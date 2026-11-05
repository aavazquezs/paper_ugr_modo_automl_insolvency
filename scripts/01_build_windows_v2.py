#!/usr/bin/env python3
"""Build firm-year windows from raw_data.csv (unbalanced, no preprocessing).

Design decisions:
- One window per (CIF, t) with 10 consecutive observed years t-9..t.
- Y = 1 if the firm becomes insolvent in (t, t+h]; 0 otherwise.
- Firms with Type=Solvent have Y=NaN in raw: they count as 0 (never insolvent).
- Windows with t >= failure year are discarded (they would see the failure).
- Negatives require the firm to be observed up to t+h (known status).
- F27, F30, F42, F43 flip sign in 2008 for all firms (PGC 2007 change):
  their pre-2008 values are multiplied by -1.
- FCA is dropped (its values are later than the observation year).
- N1 (firm age) is dropped: for 28.9% of solvent firms it counts from the start of the
  panel (N1=0 in 1999) and resets at the founding date; this never happens for
  insolvent firms, so N1 identifies the data source rather than the age.
- N11 (manager gender) is dropped: it is missing in 68% of rows of insolvent firms
  vs 8% of solvent firms, so its missingness encodes the label.
- Firms with Type=Insolvent but no Y=1 year (50,563) are KEPT as negatives (same rule:
  observed with Y=0 up to t+h). Excluding them would select the sample using the final
  outcome (Type), leaving only never-failing firms as negatives and inflating metrics.
- See RESULTADOS_EJECUCIONES.md (section 6) for the evidence.
- Lags: <var>_1 is year t (most recent), <var>_10 is year t-9 (oldest).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
RAW = Path('/media/datos/WORKSPACE/DOCTORADO/datasets/raw_data.csv')
OUTDIR = ROOT / 'data' / 'windows'
OUTDIR.mkdir(parents=True, exist_ok=True)

WINDOW = 10
HORIZONS = [1, 3, 5]
SIGN_BREAK_VARS = ['F27', 'F30', 'F42', 'F43']
SIGN_BREAK_YEAR = 2008
STATIC_VARS = ['N3', 'N4', 'N5', 'N8', 'N15']
DROP_VARS = ['FCA', 'N1', 'N11']
EXCLUDE_UNDATED_INSOLVENT = False  # True = outcome-based selection (biased), see docstring
VERSION = 'v3'


def main():
    print('Loading raw...')
    df = pd.read_csv(RAW)
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed') or c == ''] + DROP_VARS)
    df['CIF'] = df['CIF'].astype(str)
    assert not df.duplicated(['CIF', 'Year']).any(), 'Duplicated CIF-Year rows'
    if EXCLUDE_UNDATED_INSOLVENT:
        dated = set(df.loc[df['Y'] == 1, 'CIF'])
        undated = (df['Type'] == 'Insolvent') & ~df['CIF'].isin(dated)
        print(f'  excluding {df.loc[undated, "CIF"].nunique()} insolvent firms without failure year')
        df = df[~undated]

    # Sign correction for the 2008 accounting break
    pre = df['Year'] < SIGN_BREAK_YEAR
    for v in SIGN_BREAK_VARS:
        before = (df.loc[pre, v] < 0).mean()
        df.loc[pre, v] = -df.loc[pre, v]
        print(f'  {v}: neg% pre-2008 {before:.2f} -> {(df.loc[pre, v] < 0).mean():.2f}, '
              f'post-2008 {(df.loc[~pre, v] < 0).mean():.2f}')

    time_vars = [c for c in df.columns if c[0] in 'NF' and c not in STATIC_VARS]
    y0, y1 = df['Year'].min(), df['Year'].max()
    n_years = y1 - y0 + 1

    # Firm-level info
    cifs, firm_idx = np.unique(df['CIF'].values, return_inverse=True)
    year_idx = (df['Year'].values - y0).astype(int)
    n_firms = len(cifs)
    fail = df[df['Y'] == 1].groupby('CIF')['Year'].min()
    assert (df[df['Y'] == 1].groupby('CIF').size() == 1).all()
    fail_year = pd.Series(cifs).map(fail).fillna(np.inf).values
    last_year = df.groupby('CIF')['Year'].max().reindex(cifs).values

    # Panel [firm, year]
    present = np.zeros((n_firms, n_years), dtype=bool)
    present[firm_idx, year_idx] = True
    panel = {}
    for v in time_vars:
        if df[v].dtype.kind in 'fiub':
            a = np.full((n_firms, n_years), np.nan, dtype=np.float32)
            a[firm_idx, year_idx] = df[v].values.astype(np.float32)
        else:  # time-varying categoricals (N9, N10, N11)
            a = np.full((n_firms, n_years), None, dtype=object)
            a[firm_idx, year_idx] = df[v].values
        panel[v] = a
    static = {}
    for v in STATIC_VARS:
        a = pd.DataFrame(np.full((n_firms, n_years), None, dtype=object))
        a.values[firm_idx, year_idx] = df[v].values
        static[v] = a.ffill(axis=1).values  # most recent non-null up to t
    del df

    # Window validity: 10 consecutive observed years ending at t
    csum = np.cumsum(np.pad(present, ((0, 0), (1, 0))), axis=1)
    full = np.zeros_like(present)
    full[:, WINDOW - 1:] = (csum[:, WINDOW:] - csum[:, :-WINDOW]) == WINDOW

    for h in HORIZONS:
        f_i, t_i = np.nonzero(full)
        t_year = t_i + y0
        fy = fail_year[f_i]
        # same censoring for both classes: the whole horizon must fall within the data
        before_fail = (t_year < fy) & (t_year + h <= y1)
        pos = before_fail & (fy <= t_year + h)
        neg = before_fail & (fy > t_year + h) & (t_year + h <= last_year[f_i])
        keep = pos | neg
        f_i, t_i, t_year, yv = f_i[keep], t_i[keep], t_year[keep], pos[keep].astype(np.int8)

        cols = {
            'CIF': cifs[f_i],
            'window_id': [f'{c}_{t}_h{h}' for c, t in zip(cifs[f_i], t_year)],
            'ref_year': t_year,
            'horizon': np.full(len(f_i), h, dtype=np.int8),
        }
        for v in STATIC_VARS:
            cols[v] = static[v][f_i, t_i]
        for v in time_vars:
            for lag in range(1, WINDOW + 1):
                cols[f'{v}_{lag}'] = panel[v][f_i, t_i - lag + 1]
        cols['Y'] = yv
        out = pd.DataFrame(cols)
        print(f'h{h}: {out.shape}, CIFs={out["CIF"].nunique()}, Y={out["Y"].value_counts().to_dict()}, '
              f'prevalence={out["Y"].mean():.4f}')
        print('  ref_year range by Y:', out.groupby('Y')['ref_year'].agg(['min', 'median', 'max']).to_dict('index'))
        out.to_pickle(OUTDIR / f'windows_{VERSION}_h{h}_unbalanced.pkl')


if __name__ == '__main__':
    main()
