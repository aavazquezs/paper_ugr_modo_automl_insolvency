#!/usr/bin/env python3
"""Permutation feature importance (AUCPR) of the selected v3 models on the full
unbalanced test set: GBM_2 (h1), XGBoost_3 (h3, h5).

Uses the saved H2O models from runs/leakfree_v3_h{h}/ (no retraining).
"""
import glob
import os
import h2o
import pandas as pd

ROOT = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'
VERSION = 'v3'
SEED = 1234
N_REPEATS = 3
OUT = os.path.join(ROOT, 'runs', 'leakfree_v3_extra')
SELECTED = {1: 'GBM_2', 3: 'XGBoost_3', 5: 'XGBoost_3'}
os.makedirs(OUT, exist_ok=True)


def base_var(col):
    """F21_1 -> F21; N15.No (one-hot) -> N15; static vars unchanged."""
    col = col.split('.')[0]
    head, _, tail = col.rpartition('_')
    return head if head and tail.isdigit() else col


h2o.init(nthreads=2, max_mem_size='4G')
for h, tag in SELECTED.items():
    model_dir = glob.glob(os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}', f'{tag}_AutoML_*'))
    assert len(model_dir) == 1, model_dir
    model = h2o.load_model(model_dir[0])

    test = h2o.import_file(os.path.join(ROOT, 'data', 'final',
                                        f'test_unbalanced_leakfree_{VERSION}_h{h}.csv'))
    test['Y'] = test['Y'].asfactor()
    print(f'h{h} {tag}: test {test.nrow} rows; baseline AUCPR = '
          f'{model.model_performance(test).aucpr():.4f}', flush=True)

    pi = model.permutation_importance(test, metric='PR_AUC', n_samples=-1,
                                      n_repeats=N_REPEATS, seed=SEED, use_pandas=True)
    pi = pi.reset_index().rename(columns={'index': 'variable'})
    runs = [c for c in pi.columns if c.startswith('Run')]
    out = pd.DataFrame({'variable': pi['Variable'] if 'Variable' in pi else pi['variable'],
                        'mean_drop': pi[runs].mean(axis=1),
                        'sd_drop': pi[runs].std(axis=1)})
    out['base_var'] = out['variable'].map(base_var)
    out = out.sort_values('mean_drop', ascending=False)
    out.to_csv(os.path.join(OUT, f'permimp_h{h}_{tag}.csv'), index=False)

    agg = (out.groupby('base_var')['mean_drop'].sum().clip(lower=0)
              .sort_values(ascending=False).to_frame('sum_drop'))
    agg['pct'] = 100 * agg['sum_drop'] / agg['sum_drop'].sum()
    agg.to_csv(os.path.join(OUT, f'permimp_base_h{h}_{tag}.csv'))
    print(out.head(10).to_string(index=False), flush=True)
    h2o.remove_all()

h2o.cluster().shutdown()
