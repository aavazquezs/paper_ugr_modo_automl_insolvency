#!/usr/bin/env python3
"""Out-of-time split of the leak-free v3 windows (robustness check).

Train: windows whose label is observable at the cutoff (ref_year + h <= CUTOFF).
Test:  windows with ref_year >= CUTOFF, all firms (pure temporal split). The CIFs
       absent from train are saved so metrics can also be reported on unseen firms.
Same preprocessing as 02_split_preprocess_v2.py: drop variables with >50% NA in
train, undersample train 1:1, CV folds by CIF, test keeps natural prevalence.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
WINDOWS = ROOT / 'data' / 'windows'
OUT_SPLITS = ROOT / 'data' / 'splits'
OUT_FINAL = ROOT / 'data' / 'final'

SEED = 1234
MAX_MISSING = 0.5
NFOLDS = 3
CUTOFF = 2014
SRC_VERSION, VERSION = 'v3', 'v3oot'
META = ['CIF', 'window_id', 'ref_year', 'horizon', 'Y']


def base_var(col):
    return col.rsplit('_', 1)[0] if col.rsplit('_', 1)[-1].isdigit() else col


for h in [1, 3, 5]:
    df = pd.read_pickle(WINDOWS / f'windows_{SRC_VERSION}_h{h}_unbalanced.pkl')
    train = df[df['ref_year'] + h <= CUTOFF].copy()
    test = df[df['ref_year'] >= CUTOFF].copy()
    unseen = sorted(set(test['CIF']) - set(train['CIF']))
    unseen_set = set(unseen)
    test_unseen = test[test['CIF'].isin(unseen_set)]
    print(f'h={h} cutoff={CUTOFF}')
    print(f'  train: {len(train)} windows, {train["CIF"].nunique()} CIFs, '
          f'ref_year {train["ref_year"].min()}-{train["ref_year"].max()}, pos={int(train["Y"].sum())}')
    print(f'  test:  {len(test)} windows, {test["CIF"].nunique()} CIFs, '
          f'ref_year {test["ref_year"].min()}-{test["ref_year"].max()}, pos={int(test["Y"].sum())}, '
          f'prev={test["Y"].mean():.4f}')
    print(f'  test unseen firms: {len(test_unseen)} windows, {len(unseen)} CIFs, '
          f'pos={int(test_unseen["Y"].sum())}')
    with open(OUT_SPLITS / f'test_unseen_cifs_{VERSION}_h{h}.json', 'w') as f:
        json.dump(unseen, f)

    feats = [c for c in df.columns if c not in META]
    miss = train[feats].isna().mean().groupby(base_var).mean()
    dropped = miss[miss > MAX_MISSING].index.tolist()
    print(f'  dropped (train missing > {MAX_MISSING:.0%}): {dict(miss[dropped].round(2))}')
    drop_cols = [c for c in feats if base_var(c) in dropped]
    train, test = train.drop(columns=drop_cols), test.drop(columns=drop_cols)

    pos = train[train['Y'] == 1]
    neg = train[train['Y'] == 0].sample(n=len(pos), random_state=SEED)
    train_bal = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    cif_list = train_bal['CIF'].unique()
    rng = np.random.default_rng(SEED)
    train_bal['fold'] = train_bal['CIF'].map(dict(zip(cif_list, rng.integers(0, NFOLDS, len(cif_list)))))

    print(f'  train_balanced: {len(train_bal)} rows, cols={train_bal.shape[1]}; test cols={test.shape[1]}')
    train_bal.to_csv(OUT_FINAL / f'train_balanced_leakfree_{VERSION}_h{h}.csv', index=False)
    test.to_csv(OUT_FINAL / f'test_unbalanced_leakfree_{VERSION}_h{h}.csv', index=False)
