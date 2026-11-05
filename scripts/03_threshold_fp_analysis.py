#!/usr/bin/env python3
"""Threshold chosen without the test set + analysis of the leader's false positives.

1. Threshold: the leader's cross-validation threshold table (balanced 1:1 train, folds by CIF)
   is reweighted to the natural train prevalence: negatives were undersampled at random with
   rate beta = n_pos / n_neg (train windows), so each CV negative stands for 1/beta negatives.
   The threshold maximising this reweighted F1 is then applied to the test set unchanged.
   For reference it also reports the threshold that maximises F1 on test (optimistic) and the
   one that maximises F1 on the balanced CV (ignores prevalence).
2. False positives: for each test window (CIF, t) predicted insolvent but with Y=0, looks up
   whether the firm fails later (after t+h) in raw_data.csv.

Outputs go to runs/leakfree_<VERSION>_h{h}/ (threshold CSV, FP CSV, test predictions).
"""
import glob
import json
import os

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']

import h2o
import numpy as np
import pandas as pd

ROOT = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'
RAW = '/media/datos/WORKSPACE/DOCTORADO/datasets/raw_data.csv'
VERSION = os.environ.get('LEAKFREE_VERSION', 'v3')
HORIZONS = [1, 3, 5]


def confusion(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else np.nan
    return dict(threshold=thr, TP=tp, FP=fp, FN=fn, TN=tn, recall=rec, precision=prec, f1=f1,
                alerts=tp + fp)


def main():
    raw = pd.read_csv(RAW, usecols=['CIF', 'Year', 'Y', 'Type'])
    raw['CIF'] = raw['CIF'].astype(str)
    fail_year = raw[raw['Y'] == 1].groupby('CIF')['Year'].min()
    firm_type = raw.groupby('CIF')['Type'].first()
    del raw

    h2o.init(nthreads=2, max_mem_size='4G')
    summary = []
    for h in HORIZONS:
        h2o.remove_all()
        outdir = os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}')
        lb = pd.read_csv(glob.glob(os.path.join(outdir, '*_leaderboard.csv'))[0])
        leader_id = lb['model_id'].iloc[0]
        for path in sorted(glob.glob(os.path.join(outdir, '*_AutoML_*'))):
            if os.path.isdir(path) or path.endswith('.zip'):
                continue
            if not os.path.basename(path).startswith('StackedEnsemble'):
                h2o.load_model(path)
        leader = h2o.load_model(os.path.join(outdir, leader_id))

        # Natural prevalence of the train windows (before undersampling)
        win = pd.read_pickle(os.path.join(ROOT, 'data', 'windows', f'windows_{VERSION}_h{h}_unbalanced.pkl'),
                             )[['CIF', 'ref_year', 'Y']]
        with open(os.path.join(ROOT, 'data', 'splits', f'train_cifs_{VERSION}_h{h}.json')) as f:
            train_cifs = set(json.load(f))
        tr = win[win['CIF'].isin(train_cifs)]
        beta = (tr['Y'] == 1).sum() / (tr['Y'] == 0).sum()

        # CV threshold table, reweighted to natural prevalence
        xval = leader._model_json['output']['cross_validation_metrics']
        tt = xval._metric_json['thresholds_and_metric_scores'].as_data_frame()
        fps_nat = tt['fps'] / beta
        prec_nat = tt['tps'] / (tt['tps'] + fps_nat)
        rec = tt['tps'] / (tt['tps'] + tt['fns'])
        f1_nat = 2 * prec_nat * rec / (prec_nat + rec)
        thr_cv_nat = float(tt['threshold'].iloc[int(np.nanargmax(f1_nat.values))])
        thr_cv_bal = float(tt['threshold'].iloc[int(np.nanargmax(tt['f1'].values))])

        # Test predictions
        test_path = os.path.join(ROOT, 'data', 'final', f'test_unbalanced_leakfree_{VERSION}_h{h}.csv')
        test = h2o.import_file(test_path, destination_frame=f'test_h{h}')
        test['Y'] = test['Y'].asfactor()
        pred = leader.predict(test)['p1'].as_data_frame()['p1'].values
        meta = test[['CIF', 'ref_year', 'Y']].as_data_frame()
        meta['CIF'] = meta['CIF'].astype(str)
        meta['Y'] = meta['Y'].astype(int)
        meta['p1'] = pred
        meta.to_csv(os.path.join(outdir, f'leader_test_predictions_h{h}.csv'), index=False)
        y, p = meta['Y'].values, meta['p1'].values

        thr_test = float(leader.model_performance(test).find_threshold_by_max_metric('f1'))
        rows = []
        for name, thr in [('F1 máx. en CV reponderada (sin mirar test)', thr_cv_nat),
                          ('F1 máx. en test (optimista)', thr_test),
                          ('F1 máx. en CV balanceada', thr_cv_bal)]:
            r = confusion(y, p, thr); r.update(h=f'h{h}', criterio=name, beta=beta)
            rows.append(r)
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(outdir, f'leader_thresholds_h{h}.csv'), index=False)
        print(res.to_string(index=False))
        summary.append(res)

        # False positives at the threshold chosen without test
        m = meta.copy()
        m['pred'] = m['p1'] >= thr_cv_nat
        m['fail_year'] = m['CIF'].map(fail_year)
        m['type'] = m['CIF'].map(firm_type)
        gap = m['fail_year'] - (m['ref_year'] + h)  # years after the horizon end
        m['outcome'] = np.select(
            [m['fail_year'].notna() & (gap <= 2), m['fail_year'].notna() & (gap > 2),
             m['type'] == 'Insolvent'],
            ['quiebra 1-2 años después del horizonte', 'quiebra 3+ años después del horizonte',
             'Insolvent sin año de quiebra'],
            default='Solvent')
        neg = m[m['Y'] == 0]
        tab = pd.DataFrame({
            'FP_n': neg[neg['pred']]['outcome'].value_counts(),
            'FP_pct': neg[neg['pred']]['outcome'].value_counts(normalize=True) * 100,
            'TN_pct': neg[~neg['pred']]['outcome'].value_counts(normalize=True) * 100,
        }).fillna(0)
        tab['FP_firms'] = neg[neg['pred']].groupby('outcome')['CIF'].nunique()
        tab['alert_rate_pct'] = neg.groupby('outcome')['pred'].mean() * 100
        tab.to_csv(os.path.join(outdir, f'leader_false_positives_h{h}.csv'))
        print(tab.round(2).to_string())

    pd.concat(summary).to_csv(os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_thresholds_summary.csv'),
                              index=False)
    h2o.cluster().shutdown(prompt=False)


if __name__ == '__main__':
    main()
