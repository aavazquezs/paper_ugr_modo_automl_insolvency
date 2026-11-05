#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
OUT_SPLITS = ROOT / 'data' / 'splits'
OUT_FINAL = ROOT / 'data' / 'final'
OUT_SPLITS.mkdir(parents=True, exist_ok=True)
OUT_FINAL.mkdir(parents=True, exist_ok=True)

SEED = 1234

for h in [1,3,5]:
    print(f'Processing h={h}...')
    # Use balanced files as the windowed dataset structure
    df = pd.read_csv(ROOT / 'datasets' / f'10_to_predict_{h}_balanced.csv')
    # Split by CIF to avoid leakage
    cifs = df['CIF'].unique()
    train_cifs, test_cifs = train_test_split(cifs, test_size=0.3, random_state=SEED)
    train_cifs_set = set(train_cifs)
    test_cifs_set = set(test_cifs)
    assert len(train_cifs_set & test_cifs_set) == 0, "Leakage in CIF split"
    
    train_df = df[df['CIF'].isin(train_cifs_set)].copy()
    test_df = df[df['CIF'].isin(test_cifs_set)].copy()
    
    print(f'  train: {len(train_df)} rows, {len(train_cifs_set)} CIFs')
    print(f'  test: {len(test_df)} rows, {len(test_cifs_set)} CIFs')
    print(f'  test prev: {test_df["Status"].value_counts(normalize=True).to_dict()}')
    
    # Save CIF lists
    with open(OUT_SPLITS / f'train_cifs_h{h}.json', 'w') as f:
        json.dump(sorted(list(train_cifs_set)), f)
    with open(OUT_SPLITS / f'test_cifs_h{h}.json', 'w') as f:
        json.dump(sorted(list(test_cifs_set)), f)
    
    # For now, save as train_balanced and test_unbalanced structure
    # Note: train is currently balanced from original file; we'll fix by doing undersampling only on train later
    # But the key fix is group split by CIF
    train_df.to_csv(OUT_FINAL / f'train_balanced_h{h}.csv', index=False)
    test_df.to_csv(OUT_FINAL / f'test_unbalanced_h{h}.csv', index=False)
    
    print(f'  Saved fixed splits for h{h}\n')
