#!/usr/bin/env python3
"""Figures for the manuscript from the leak-free v3 results.

pareto: cross-validated AUCPR (the AutoML sort metric used for model selection) vs
prediction time per row, one panel per horizon (a/b/c), markers by algorithm family,
Pareto front and selected model.
Usage: 08_paper_figures.py <output_dir>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'
VERSION = 'v3'
OUT = sys.argv[1]
HORIZONS = [1, 3, 5]
SELECTED = {1: 'GBM_2', 3: 'XGBoost_3', 5: 'XGBoost_3'}
FAMILIES = {  # label: (color, marker), same palette as the original figure
    'StackedEnsemble': ('red', 'o'), 'XGBoost': ('blue', '^'), 'GBM': ('green', 's'),
    'DRF': ('orange', 'o'), 'XRT': ('saddlebrown', 'v'), 'GLM': ('magenta', 'D'),
}


def family(model):
    return model.split('_')[0]


def pareto_front(df, x, y):
    """Models not dominated by any other one (lower x and higher y are better)."""
    front, best = [], float('-inf')
    for _, row in df.sort_values([x, y], ascending=[True, False]).iterrows():
        if row[y] > best:
            front.append(row)
            best = row[y]
    return pd.DataFrame(front)


def load(h):
    d = os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}')
    lb = pd.read_csv(os.path.join(d, f'10_to_predict_{h}_leakfree_leaderboard.csv'))
    tm = pd.read_csv(os.path.join(d, f'10_to_predict_{h}_leakfree_test_metrics.csv'))
    df = lb[['model_id', 'aucpr', 'predict_time_per_row_ms']].merge(tm[['model_id', 'test_aucpr']], on='model_id')
    df['model'] = df['model_id'].str.replace(r'_AutoML.*', '', regex=True)
    df['family'] = df['model'].map(family)
    return df


def fig_pareto():
    x, y = 'predict_time_per_row_ms', 'aucpr'
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(6.5, 13))
    for ax, h, letter in zip(axes, HORIZONS, 'abc'):
        df = load(h)
        front = pareto_front(df, x, y)
        ax.plot(front[x], front[y], color='red', linewidth=1.2, label='Pareto front')
        for fam, (color, marker) in FAMILIES.items():
            g = df[df['family'] == fam]
            if len(g):
                ax.scatter(g[x], g[y], color=color, marker=marker, alpha=0.7, s=30, label=fam)
        sel = df[df['model'] == SELECTED[h]].iloc[0]
        ax.scatter(sel[x], sel[y], s=160, facecolors='none', edgecolors='black', linewidths=1.2,
                   label='Selected model')
        ax.annotate(SELECTED[h], (sel[x], sel[y]), xytext=(8, -12), textcoords='offset points', fontsize=8)
        ax.set_title(f'{letter}) h = {h}')
        ax.set_xlabel('Prediction time per row (ms)')
        ax.set_ylabel('AUCPR (cross-validation)')
        ax.grid(True, alpha=0.6)
        ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pareto_front_leakfree.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_pareto()
