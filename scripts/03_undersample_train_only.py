#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from sklearn.utils import resample

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
OUT_FINAL = ROOT / 'data' / 'final'
SEED = 1234

for h in [1,3,5]:
    print(f'h={h}')
    # Load train from our fixed split (grouped by CIF) - train comes from balanced file but split correctly
    train = pd.read_csv(OUT_FINAL / f'train_balanced_h{h}.csv')
    test = pd.read_csv(OUT_FINAL / f'test_unbalanced_h{h}.csv')
    print(f'  train orig: {len(train)}, test unbal: {len(test)}, test prev: {test["Y"].value_counts(normalize=True).to_dict()}')
    # Undersample train only
    # Find target column name
    target = 'Status' if 'Status' in train.columns else 'Y'
    if target == 'Status':
        train[target] = train[target].map({'Solvent': 0, 'Insolvent': 1}) if train[target].dtype == object else train[target]
        test[target] = test[target].map({'Solvent': 0, 'Insolvent': 1}) if test[target].dtype == object else test[target]
    # balance train
    minority = train[train[target]==1]
    majority = train[train[target]==0]
    majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=SEED)
    train_bal = pd.concat([majority_down, minority]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f'  train_balanced: {len(train_bal)}, class dist: {train_bal[target].value_counts().to_dict()}')
    # ensure test unchanged
    train_bal.to_csv(OUT_FINAL / f'train_balanced_leakfree_h{h}.csv', index=False)
    train_bal.to_parquet(OUT_FINAL / f'train_balanced_leakfree_h{h}.parquet', index=False)
    test.to_csv(OUT_FINAL / f'test_unbalanced_leakfree_h{h}.csv', index=False)
    test.to_parquet(OUT_FINAL / f'test_unbalanced_leakfree_h{h}.parquet', index=False)
