#!/usr/bin/env python3
"""Composition of the leak-free v3 window datasets (for the paper's dataset table).

For each horizon h and subset (all/train/test) reports windows, firms, positives,
prevalence, and the distribution by sector (N3) and size (N8), counted both
over windows and over firms (last available value per CIF).
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
WINDOWS = ROOT / 'data' / 'windows'
SPLITS = ROOT / 'data' / 'splits'
OUT = ROOT / 'runs' / 'leakfree_v3_extra'
OUT.mkdir(parents=True, exist_ok=True)

VERSION = 'v3'
HORIZONS = [1, 3, 5]
CATS = {'N3': 'Sector', 'N8': 'Size'}


def summarise(df, h, subset):
    firms = df.sort_values('ref_year').groupby('CIF').last()
    rows = [
        dict(h=h, subset=subset, block='Total', level='Windows', n=len(df), pct=100.0),
        dict(h=h, subset=subset, block='Total', level='Firms', n=len(firms), pct=100.0),
        dict(h=h, subset=subset, block='Total', level='Positive windows',
             n=int(df['Y'].sum()), pct=100 * df['Y'].mean()),
        dict(h=h, subset=subset, block='Total', level='Firms with a positive window',
             n=int(df.groupby('CIF')['Y'].max().sum()),
             pct=100 * df.groupby('CIF')['Y'].max().mean()),
    ]
    for col, name in CATS.items():
        for unit, data in (('windows', df), ('firms', firms)):
            counts = data[col].value_counts(dropna=False)
            for level, n in counts.items():
                rows.append(dict(h=h, subset=subset, block=f'{name} ({unit})',
                                 level=str(level), n=int(n), pct=100 * n / len(data)))
    return rows


md = ['# Composition of the leak-free v3 datasets', '']
for h in HORIZONS:
    df = pd.read_pickle(WINDOWS / f'windows_{VERSION}_h{h}_unbalanced.pkl')
    train_cifs = set(json.load(open(SPLITS / f'train_cifs_{VERSION}_h{h}.json')))
    test_cifs = set(json.load(open(SPLITS / f'test_cifs_{VERSION}_h{h}.json')))
    assert not train_cifs & test_cifs
    rows = []
    for subset, part in (('all', df),
                         ('train', df[df['CIF'].isin(train_cifs)]),
                         ('test', df[df['CIF'].isin(test_cifs)])):
        rows += summarise(part, h, subset)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f'composition_h{h}.csv', index=False)

    wide = res.pivot_table(index=['block', 'level'], columns='subset',
                           values=['n', 'pct'], sort=False)
    wide.columns = [f'{s} {m}' for m, s in wide.columns]
    wide = wide[[f'{s} {m}' for s in ('all', 'train', 'test') for m in ('n', 'pct')]]
    md += [f'## h{h}', '', wide.round(2).to_markdown(), '']
    print(f'h{h}: {len(df)} windows, {df["CIF"].nunique()} firms')

(OUT / 'composition_summary.md').write_text('\n'.join(md))
print('written to', OUT)
