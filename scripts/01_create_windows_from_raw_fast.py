#!/usr/bin/env python3
import os
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
    if '' in df.columns:
        df = df.drop(columns=[''])
    df['CIF'] = df['CIF'].astype(str)
    df = df.sort_values(['CIF', 'Year'])
    
    var_cols = [c for c in df.columns if c.startswith(('N','F'))]
    static_like = ['N3','N4','N5','N8','N15']
    time_vars = [c for c in var_cols if c not in static_like]
    
    horizons = [1,3,5]
    window_size = 10
    
    for h in horizons:
        print(f'Processing horizon {h}...')
        results = []
        # process per CIF more efficiently
        for cif, g in df.groupby('CIF', sort=False):
            g = g.sort_values('Year')
            if len(g) < window_size:
                continue
            years = g['Year'].values
            yvals = g['Y'].values
            # build arrays
            # try windows
            for i in range(window_size-1, len(g)):
                end_year = int(years[i])
                # look ahead
                # search
                # faster: check if end_year+h exists
                # build dict of year->y
                pass  # skip slow way
        print('done')
        break

if __name__ == '__main__':
    main()
