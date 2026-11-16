#!/usr/bin/env python3
"""Figures for the manuscript from the leak-free v3 results.

pareto: cross-validated AUCPR (the AutoML sort metric used for model selection) vs
prediction time per row, one panel per horizon (a/b/c), markers by algorithm family,
Pareto front and selected model.
varimp: importance of the selected models (GBM_2 h1, XGBoost_3 h3/h5), one row per
horizon: (left) H2O native scaled importance, (right) permutation importance on the
unbalanced test set (mean drop in AUCPR, error bar = SD over 3 repeats). Top 10 columns.
shap: H2O SHAP summary plots of the selected models, copied with the title band
(internal model id) cropped.
Usage: 08_paper_figures.py <output_dir> [pareto|varimp|shap ...]   (default: all)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

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


TOP_N = 10


def fig_varimp():
    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(10, 11))
    for row, h, letter in zip(axes, HORIZONS, 'abc'):
        tag = SELECTED[h]
        native = pd.read_csv(os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}',
                                          f'10_to_predict_{h}_leakfree_{tag}_varimp.csv')).head(TOP_N)
        perm = pd.read_csv(os.path.join(ROOT, 'runs', 'leakfree_v3_extra',
                                        f'permimp_h{h}_{tag}.csv')).head(TOP_N)
        ax = row[0]
        ax.barh(native['variable'][::-1], native['scaled_importance'][::-1], color='tab:blue')
        ax.set_title(f'{letter}) h = {h}, {tag}: H2O importance')
        ax.set_xlabel('Scaled importance')
        ax = row[1]
        ax.barh(perm['variable'][::-1], perm['mean_drop'][::-1], xerr=perm['sd_drop'][::-1],
                color='tab:orange', capsize=2)
        ax.set_title(f'{letter}) h = {h}, {tag}: permutation importance')
        ax.set_xlabel('Mean decrease in AUCPR (test)')
        for ax in row:
            ax.grid(True, axis='x', alpha=0.6)
            ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'varimp_leakfree.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_shap():
    for h in HORIZONS:
        src = os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}', 'plots', f'{SELECTED[h]}_shap_summary.png')
        img = np.asarray(Image.open(src).convert('RGB')).copy()
        dark = img.mean(axis=2) < 100
        # top spine of the axes: first row where most pixels are dark
        spine = int(np.where(dark.mean(axis=1) > 0.5)[0][0])
        cols = np.where(dark[spine])[0]
        right = cols[np.argmax(np.diff(cols) > 1)] if (np.diff(cols) > 1).any() else cols[-1]
        img[:spine, cols[0]:right + 1] = 255  # blank the title above the main axes (not the colorbar)
        content = np.where((img.mean(axis=2) < 250).any(axis=1))[0]
        top = max(int(content[0]) - 6, 0)  # keep what lies outside the axes (colorbar label)
        Image.fromarray(img[top:]).save(os.path.join(OUT, f'shap_summary_leakfree_h{h}.png'), optimize=True)


FIGURES = {'pareto': fig_pareto, 'varimp': fig_varimp, 'shap': fig_shap}

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[2:] or FIGURES:
        FIGURES[name]()
