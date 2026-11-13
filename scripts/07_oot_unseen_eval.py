#!/usr/bin/env python3
"""Out-of-time (v3oot) test metrics by firm subset: all, seen in train, unseen.

Loads the saved v3oot H2O models (no retraining) and evaluates them on the
temporal test set split by whether the firm (CIF) has windows in train.
Usage: 07_oot_unseen_eval.py [tmp_dir]   (tmp_dir holds the temporary subset CSVs)
"""
import glob
import json
import os
import sys

import h2o
import pandas as pd

ROOT = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'
VERSION = 'v3oot'
OUT = os.path.join(ROOT, 'runs', 'leakfree_v3_extra')
TMP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, 'tmp_oot_subsets')
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']
h2o.init(nthreads=2, max_mem_size='4G')

rows = []
for h in [1, 3, 5]:
    run_dir = os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}')
    model_dirs = sorted(d for d in glob.glob(os.path.join(run_dir, '*_AutoML_*')) if not d.endswith('.zip'))
    model_dirs.sort(key=lambda d: os.path.basename(d).startswith('StackedEnsemble'))  # base models first
    models = {os.path.basename(d).split('_AutoML_')[0]: h2o.load_model(d) for d in model_dirs}
    assert len(models) == 12, list(models)

    test = pd.read_csv(os.path.join(ROOT, 'data', 'final', f'test_unbalanced_leakfree_{VERSION}_h{h}.csv'),
                       low_memory=False)
    with open(os.path.join(ROOT, 'data', 'splits', f'test_unseen_cifs_{VERSION}_h{h}.json')) as f:
        unseen = set(json.load(f))
    is_unseen = test['CIF'].isin(unseen)
    subsets = {'all': test, 'seen': test[~is_unseen], 'unseen': test[is_unseen]}

    for name, df in subsets.items():
        path = os.path.join(TMP, f'oot_h{h}_{name}.csv')
        df.to_csv(path, index=False)
        frame = h2o.import_file(path)
        frame['Y'] = frame['Y'].asfactor()
        n_pos, prev = int(df['Y'].sum()), df['Y'].mean()
        for tag, model in models.items():
            perf = model.model_performance(frame)
            rows.append({'h': h, 'model': tag, 'subset': name, 'n_windows': len(df),
                         'n_firms': df['CIF'].nunique(), 'n_pos': n_pos, 'prevalence': prev,
                         'auc': perf.auc(), 'aucpr': perf.aucpr(), 'lift': perf.aucpr() / prev})
        h2o.remove(frame)
        os.remove(path)
        print(f'h{h} {name}: {len(df)} windows, {n_pos} pos done', flush=True)
    h2o.remove_all()

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'oot_subset_metrics.csv'), index=False)
key = res[res['model'].isin(['StackedEnsemble_AllModels_1', 'GBM_2', 'XGBoost_3'])]
print(key.pivot_table(index=['h', 'model'], columns='subset', values=['auc', 'aucpr', 'lift']).round(4).to_string())
h2o.cluster().shutdown()
