#!/usr/bin/env python3
"""Split firm-year windows by CIF, preprocess using train only, undersample train only.

- Group split 70/30 by CIF, stratified on whether the firm has any positive window.
- Variables whose missing rate in train exceeds MAX_MISSING are dropped (decided on train,
  applied to test). No imputation: H2O handles missing values natively.
- Train is undersampled 1:1 (Y=0 down to the number of Y=1); test keeps natural prevalence.
- A 'fold' column (assigned per CIF) is added to train so AutoML CV never splits a firm.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
WINDOWS = ROOT / 'data' / 'windows'
OUT_SPLITS = ROOT / 'data' / 'splits'
OUT_FINAL = ROOT / 'data' / 'final'
OUT_SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 1234
TEST_SIZE = 0.3
MAX_MISSING = 0.5
NFOLDS = 3
VERSION = 'v3'
META = ['CIF', 'window_id', 'ref_year', 'horizon', 'Y']


def stratified_group_split(firms, test_size, seed):
    """Split firm ids into train/test keeping the share of firms with any positive window."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, ids in firms.groupby(firms).groups.items():
        ids = rng.permutation(np.asarray(ids))
        n_test = int(round(len(ids) * test_size))
        test.append(ids[:n_test])
        train.append(ids[n_test:])
    return np.concatenate(train), np.concatenate(test)


def base_var(col):
    return col.rsplit('_', 1)[0] if col.rsplit('_', 1)[-1].isdigit() else col


for h in [1, 3, 5]:
    print(f'h={h}')
    df = pd.read_pickle(WINDOWS / f'windows_{VERSION}_h{h}_unbalanced.pkl')

    firms = df.groupby('CIF')['Y'].max()
    train_cifs, test_cifs = stratified_group_split(firms, TEST_SIZE, SEED)
    train = df[df['CIF'].isin(set(train_cifs))].copy()
    test = df[df['CIF'].isin(set(test_cifs))].copy()
    assert not set(train['CIF']) & set(test['CIF'])
    print(f'  train: {len(train)} windows, {train["CIF"].nunique()} CIFs, prev={train["Y"].mean():.4f}')
    print(f'  test:  {len(test)} windows, {test["CIF"].nunique()} CIFs, prev={test["Y"].mean():.4f}')
    with open(OUT_SPLITS / f'train_cifs_{VERSION}_h{h}.json', 'w') as f:
        json.dump(sorted(train_cifs.tolist()), f)
    with open(OUT_SPLITS / f'test_cifs_{VERSION}_h{h}.json', 'w') as f:
        json.dump(sorted(test_cifs.tolist()), f)

    # Missingness analysis on train only
    feats = [c for c in df.columns if c not in META]
    miss = train[feats].isna().mean().groupby(base_var).mean().sort_values(ascending=False)
    dropped = miss[miss > MAX_MISSING].index.tolist()
    print(f'  dropped (train missing > {MAX_MISSING:.0%}): {dict(miss[dropped].round(2))}')
    drop_cols = [c for c in feats if base_var(c) in dropped]
    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    # Undersample train only
    pos = train[train['Y'] == 1]
    neg = train[train['Y'] == 0].sample(n=len(pos), random_state=SEED)
    train_bal = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # CV folds by CIF
    cif_list = train_bal['CIF'].unique()
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(cif_list, rng.integers(0, NFOLDS, len(cif_list))))
    train_bal['fold'] = train_bal['CIF'].map(fold_of)

    print(f'  train_balanced: {len(train_bal)} rows, {train_bal["CIF"].nunique()} CIFs, '
          f'Y={train_bal["Y"].value_counts().to_dict()}, cols={train_bal.shape[1]}')
    train_bal.to_csv(OUT_FINAL / f'train_balanced_leakfree_{VERSION}_h{h}.csv', index=False)
    test.to_csv(OUT_FINAL / f'test_unbalanced_leakfree_{VERSION}_h{h}.csv', index=False)
