#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
RAW = Path('/media/datos/WORKSPACE/DOCTORADO/datasets/raw_data.csv')
OUTDIR = ROOT / 'data' / 'windows'
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    print('Loading raw...')
    df = pd.read_csv(RAW)
    # normalize columns
    if '' in df.columns:
        df = df.drop(columns=[''])
    df = df.copy()
    df['CIF'] = df['CIF'].astype(str)
    df = df.sort_values(['CIF', 'Year'])
    
    # Identify variable types
    # N and F are financial/firm variables
    var_cols = [c for c in df.columns if c.startswith(('N','F'))]
    static_like = ['N3','N4','N5','N8','N15']  # often static/categorical in this setup
    time_vars = [c for c in var_cols if c not in static_like]
    
    horizons = [1,3,5]
    window_size = 10
    
    for h in horizons:
        print(f'Processing horizon {h}...')
        rows = []
        # group by CIF
        for cif, g in df.groupby('CIF'):
            g = g.sort_values('Year')
            years = g['Year'].values
            # need at least window_size history and be able to look ahead h
            if len(g) < window_size:
                continue
            # try to create windows - use sliding window of last 10 years ending at t, predict Y at t+h
            for i in range(window_size-1, len(g)):
                end_idx = i  # index in group
                start_idx = end_idx - window_size + 1
                window = g.iloc[start_idx:end_idx+1].copy()
                # check if we have future
                end_year = window['Year'].iloc[-1]
                # look for row at end_year + h
                future = g[g['Year'] == end_year + h]
                if future.empty:
                    continue
                y_future = future['Y'].iloc[0]
                if pd.isna(y_future):
                    continue
                rec = {}
                rec['CIF'] = cif
                rec['window_id'] = f"{cif}_{end_year}_h{h}"
                rec['ref_year'] = int(end_year)
                rec['horizon'] = h
                # add static-like vars from most recent or as in window - take from last year or mode
                for sc in static_like:
                    if sc in window.columns:
                        # take most recent non-null
                        vals = window[sc].dropna()
                        rec[sc] = vals.iloc[-1] if len(vals)>0 else window[sc].iloc[-1]
                # add lagged time vars: _1 is most recent (t), _10 is oldest in window
                for tv in time_vars:
                    # window is sorted by year ascending, so iloc[0] is oldest, iloc[-1] is newest
                    for lag in range(1, window_size+1):
                        rec[f'{tv}_{lag}'] = window[tv].iloc[window_size - lag]
                rec['Y'] = int(y_future) if not isinstance(y_future, str) else y_future
                rows.append(rec)
        out_df = pd.DataFrame(rows)
        print(f'h{h}: {out_df.shape}, CIFs={out_df["CIF"].nunique()}, Y={out_df["Y"].value_counts().to_dict()}')
        out_df.to_csv(OUTDIR / f'windows_h{h}_unbalanced.csv', index=False)
        out_df.to_parquet(OUTDIR / f'windows_h{h}_unbalanced.parquet', index=False)
        print(f'Saved windows_h{h}_unbalanced')

if __name__ == '__main__':
    main()
