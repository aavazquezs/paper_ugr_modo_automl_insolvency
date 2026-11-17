#!/usr/bin/env python3
"""Sensitivity of the selected models to N15 (family-owned firm).

N15 is constant within each firm and its share of 'Yes' depends on how long ago the firm
stopped filing (30% solvent, 10% insolvent with year, 3% insolvent without year), which
suggests it was recorded at extraction time. This script refits each selected model with
its exact hyperparameters twice on the same balanced training set and folds:
  - 'with_N15'    : same predictors as AutoML (control: checks that the refit reproduces it)
  - 'without_N15' : N15 removed
and reports CV and test AUC/AUCPR. Also writes the N15 share by firm group.
Outputs: runs/leakfree_v3_extra/n15_sensitivity.csv, n15_by_group.csv
"""
import glob
import os

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']

import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator, H2OXGBoostEstimator

ROOT = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'
VERSION = 'v3'
SEED = 1234
SELECTED = {1: 'GBM_2', 3: 'XGBoost_3', 5: 'XGBoost_3'}
OUT = os.path.join(ROOT, 'runs', 'leakfree_v3_extra')
NON_PREDICTORS = ['Y', 'CIF', 'window_id', 'ref_year', 'horizon', 'fold']
# parameters that refer to frames/columns or CV bookkeeping, set explicitly below
SKIP = {'model_id', 'training_frame', 'validation_frame', 'response_column', 'ignored_columns',
        'fold_column', 'nfolds', 'fold_assignment', 'weights_column', 'offset_column',
        'checkpoint', 'export_checkpoints_dir', 'keep_cross_validation_models',
        'keep_cross_validation_predictions', 'keep_cross_validation_fold_assignment',
        'calibration_frame', 'monotone_constraints', 'interaction_constraints'}


def n15_by_group():
    d = pd.read_csv('/media/datos/WORKSPACE/DOCTORADO/datasets/raw_data.csv',
                    usecols=['CIF', 'Type', 'N15', 'Y', 'Year'], low_memory=False)
    f = d.groupby('CIF').agg(Type=('Type', 'first'), N15=('N15', 'first'),
                             has_year=('Y', lambda s: (s == 1).any()), last_year=('Year', 'max'))
    f['group'] = f['Type']
    f.loc[(f.Type == 'Insolvent') & f.has_year, 'group'] = 'Insolvent with year'
    f.loc[(f.Type == 'Insolvent') & ~f.has_year, 'group'] = 'Insolvent without year'
    res = f.groupby('group').agg(firms=('N15', 'size'),
                                 n15_yes_pct=('N15', lambda s: 100 * (s == 'Yes').mean()),
                                 median_last_year=('last_year', 'median'))
    res.to_csv(os.path.join(OUT, 'n15_by_group.csv'))
    print(res.round(1))


def refit(model, x, train, test, h, variant):
    algo = model.algo
    params = {k: v['actual'] for k, v in model.params.items()
              if k not in SKIP and v['actual'] is not None and not isinstance(v['actual'], dict)}
    params.update(seed=SEED, model_id=f'{SELECTED[h]}_h{h}_{variant}')
    Est = H2OGradientBoostingEstimator if algo == 'gbm' else H2OXGBoostEstimator
    est = Est(**params)
    est.train(x=x, y='Y', training_frame=train, fold_column='fold')
    cv = est.model_performance(xval=True)
    te = est.model_performance(test)
    prev = test['Y'].asnumeric().mean()[0]
    return dict(h=h, model=SELECTED[h], variant=variant, n_predictors=len(x),
                cv_auc=cv.auc(), cv_aucpr=cv.aucpr(), test_auc=te.auc(), test_aucpr=te.aucpr(),
                test_prevalence=prev, test_lift=te.aucpr() / prev)


def main():
    n15_by_group()
    h2o.init(nthreads=2, max_mem_size='4G')
    rows = []
    for h, tag in SELECTED.items():
        train = h2o.import_file(os.path.join(ROOT, 'data', 'final', f'train_balanced_leakfree_{VERSION}_h{h}.csv'))
        test = h2o.import_file(os.path.join(ROOT, 'data', 'final', f'test_unbalanced_leakfree_{VERSION}_h{h}.csv'))
        train['Y'] = train['Y'].asfactor()
        test['Y'] = test['Y'].asfactor()
        path = [p for p in glob.glob(os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}', f'{tag}_AutoML_*'))
                if not p.endswith('.zip')][0]
        model = h2o.load_model(path)
        te = model.model_performance(test)
        rows.append(dict(h=h, model=tag, variant='automl_original', n_predictors=None,
                         cv_auc=model.auc(xval=True), cv_aucpr=model.aucpr(xval=True),
                         test_auc=te.auc(), test_aucpr=te.aucpr()))
        x_all = [c for c in train.columns if c not in NON_PREDICTORS]
        x_no = [c for c in x_all if c != 'N15']
        assert len(x_all) - len(x_no) == 1, 'N15 should be a single column'
        rows.append(refit(model, x_all, train, test, h, 'with_N15'))
        rows.append(refit(model, x_no, train, test, h, 'without_N15'))
        h2o.remove_all()
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'n15_sensitivity.csv'), index=False)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    h2o.cluster().shutdown()


if __name__ == '__main__':
    main()
