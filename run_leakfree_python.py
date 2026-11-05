#!/usr/bin/env python3
"""Leak-free H2O AutoML per horizon, with test evaluation, Pareto fronts and explanation plots.

Reproduces every analysis of the 2023 notebooks (H10h*.ipynb) and adds the new ones:
- leaderboard with training/prediction times (extra_columns="ALL")
- metrics of every model on the unbalanced test set + leader confusion matrix
- Pareto fronts: AUC (CV) vs predict time (as in 2023), AUCPR (test) vs predict time
  and AUCPR (test) vs training time
- leader learning curve, variable-importance heatmap, model correlation, metalearner coefficients
- best tree model: variable importance, SHAP summary, SHAP of one insolvent firm, learning
  curve and partial dependence of its top variables
Outputs go to runs/leakfree_<VERSION>_h{h}/ (plots in plots/).
"""
import json
import os
import subprocess
import time

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h2o
import pandas as pd
from h2o.automl import H2OAutoML, get_leaderboard

ROOT = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'
VERSION = os.environ.get('LEAKFREE_VERSION', 'v3')
HORIZONS = [int(h) for h in os.environ.get('LEAKFREE_HORIZONS', '1,3,5').split(',')]
MAX_MODELS = int(os.environ.get('LEAKFREE_MAX_MODELS', '10'))
MAX_RUNTIME_SECS = int(os.environ.get('LEAKFREE_MAX_RUNTIME_SECS', '1800'))
EXPLAIN_SAMPLE_ROWS = 10000
N_PDP = 5
SEED = 1234
TREE_ALGOS = ('GBM', 'XGBoost', 'DRF')
ALGO_COLORS = {'StackedEnsemble': 'red', 'XGBoost': 'blue', 'GBM': 'green', 'DRF': 'orange',
               'XRT': 'brown', 'DeepLearning': 'purple', 'GLM': 'gold'}


def save_plot(fn, path):
    """Run an H2O plotting call and save its figure; failures are logged, not fatal."""
    plt.close('all')
    try:
        res = fn()
        fig = res.figure() if hasattr(res, 'figure') else plt.gcf()
        fig.savefig(path, dpi=120, bbox_inches='tight')
    except Exception as e:
        print(f"Plot {os.path.basename(path)} failed: {e}")
    finally:
        plt.close('all')


def pareto_front(df, x, y):
    """Models not dominated by any other one (lower x and higher y are better)."""
    front, best = [], float('-inf')
    for _, row in df.sort_values([x, y], ascending=[True, False]).iterrows():
        if row[y] > best:
            front.append(row)
            best = row[y]
    return pd.DataFrame(front)


def plot_pareto(df, x, y, xlabel, ylabel, title, path):
    front = pareto_front(df, x, y)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, g in df.groupby('algo'):
        ax.scatter(g[x], g[y], label=algo, color=ALGO_COLORS.get(algo, 'gray'), alpha=0.8)
    ax.plot(front[x], front[y], color='black', linestyle='--', linewidth=1, label='Pareto front')
    for _, r in front.iterrows():
        ax.annotate(r['model_id'].split('_AutoML')[0], (r[x], r[y]), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return front


def run_horizon(h):
    print(f"=== Starting leak-free {VERSION} run for horizon h{h} ===")
    subprocess.run(['pkill', '-9', '-f', 'h2o.jar'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(15)

    try:
        h2o.cluster().shutdown(prompt=False)
        time.sleep(5)
    except Exception:
        pass

    outdir = os.path.join(ROOT, 'runs', f'leakfree_{VERSION}_h{h}')
    plotdir = os.path.join(outdir, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    os.chdir(outdir)

    print(f"Initializing H2O with Java 17, output dir: {outdir}")
    h2o.init(nthreads=2, max_mem_size="4G")

    train_path = os.path.join(ROOT, 'data', 'final', f'train_balanced_leakfree_{VERSION}_h{h}.csv')
    test_path = os.path.join(ROOT, 'data', 'final', f'test_unbalanced_leakfree_{VERSION}_h{h}.csv')

    print("Importing data...")
    train = h2o.import_file(train_path, destination_frame=f"train_h{h}")
    test = h2o.import_file(test_path, destination_frame=f"test_h{h}")

    y = "Y"
    x = list(train.columns)
    if y in x:
        x.remove(y)
    for c in ["CIF", "window_id", "ref_year", "horizon", "fold"]:
        if c in x:
            x.remove(c)

    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    print(f"Training AutoML for h{h}...")
    aml = H2OAutoML(
        max_models=MAX_MODELS,
        seed=SEED,
        sort_metric="AUCPR",
        max_runtime_secs=MAX_RUNTIME_SECS,
        exclude_algos=["DeepLearning"],
        keep_cross_validation_models=False,
        keep_cross_validation_predictions=False,
        keep_cross_validation_fold_assignment=False
    )
    aml.train(x=x, y=y, training_frame=train, fold_column="fold")

    print("Saving results...")
    name = f"10_to_predict_{h}_leakfree"
    df = get_leaderboard(aml, extra_columns="ALL").as_data_frame()
    df.to_csv(os.path.join(outdir, f'{name}_leaderboard.csv'), index=False)

    test_metrics = []
    for model_id in df["model_id"].tolist():
        try:
            model = h2o.get_model(model_id)
            perf = model.model_performance(test)
            test_metrics.append({"model_id": model_id, "test_auc": perf.auc(), "test_aucpr": perf.aucpr(),
                                 "test_logloss": perf.logloss(), "test_f1": perf.F1()[0][1]})
            h2o.save_model(model=model, path=outdir, force=True)
        except Exception as e:
            print(f"Error evaluating/saving model {model_id}: {e}")
    tm = pd.DataFrame(test_metrics)
    tm.to_csv(os.path.join(outdir, f'{name}_test_metrics.csv'), index=False)
    print(tm.head(5).to_string(index=False))

    try:
        aml.leader.download_mojo(path=outdir)
    except Exception as e:
        print(f"Error saving MOJO: {e}")

    # Leader confusion matrix on test (threshold that maximises F1)
    try:
        cm = aml.leader.model_performance(test).confusion_matrix()
        cm.table.as_data_frame().to_csv(os.path.join(outdir, f'{name}_leader_confusion_matrix_test.csv'), index=False)
        print(cm)
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")

    # Pareto fronts: performance vs processing speed
    perf_df = df.merge(tm, on="model_id")
    fronts = {
        'auc_cv_vs_predict_time': ('predict_time_per_row_ms', 'auc', 'AUC (CV)'),
        'aucpr_test_vs_predict_time': ('predict_time_per_row_ms', 'test_aucpr', 'AUCPR (test)'),
        'aucpr_test_vs_training_time': ('training_time_ms', 'test_aucpr', 'AUCPR (test)'),
    }
    for key, (xm, ym, ylabel) in fronts.items():
        try:
            front = plot_pareto(perf_df, xm, ym, f'{xm} (log)', ylabel, f'h{h}: {ylabel} vs {xm}',
                                os.path.join(plotdir, f'pareto_{key}.png'))
            front.to_csv(os.path.join(outdir, f'{name}_pareto_{key}.csv'), index=False)
        except Exception as e:
            print(f"Error computing Pareto front {key}: {e}")

    # Explanation plots on a random test sample (as aml.explain(test) in 2023)
    sample = test[test.runif(seed=SEED) < min(1.0, EXPLAIN_SAMPLE_ROWS / test.nrow), :]
    leader = aml.leader
    save_plot(lambda: leader.learning_curve_plot(), os.path.join(plotdir, 'leader_learning_curve.png'))
    save_plot(lambda: aml.varimp_heatmap(), os.path.join(plotdir, 'varimp_heatmap.png'))
    save_plot(lambda: aml.model_correlation_heatmap(sample), os.path.join(plotdir, 'model_correlation.png'))

    se_ids = sorted((m for m in df["model_id"] if m.startswith("StackedEnsemble")),
                    key=lambda m: not m.startswith("StackedEnsemble_AllModels"))
    if se_ids:
        try:
            metalearner = h2o.get_model(se_ids[0]).metalearner()
            with open(os.path.join(outdir, f'{name}_metalearner_coef_norm.json'), 'w') as f:
                json.dump(metalearner.coef_norm(), f, indent=2)
            save_plot(lambda: metalearner.std_coef_plot(), os.path.join(plotdir, 'metalearner_std_coef.png'))
        except Exception as e:
            print(f"Error with metalearner: {e}")

    tree_ids = [m for m in df["model_id"] if m.startswith(TREE_ALGOS)]
    if tree_ids:
        tree = h2o.get_model(tree_ids[0])
        tag = tree_ids[0].split('_AutoML')[0]
        print(f"Explaining best tree model {tag}")
        vi = tree.varimp(use_pandas=True)
        vi.to_csv(os.path.join(outdir, f'{name}_{tag}_varimp.csv'), index=False)
        save_plot(lambda: tree.varimp_plot(num_of_features=20), os.path.join(plotdir, f'{tag}_varimp.png'))
        save_plot(lambda: tree.learning_curve_plot(), os.path.join(plotdir, f'{tag}_learning_curve.png'))
        save_plot(lambda: tree.shap_summary_plot(sample), os.path.join(plotdir, f'{tag}_shap_summary.png'))
        positives = sample[y].as_data_frame()[y].astype(str).tolist()
        if "1" in positives:
            row = positives.index("1")
            save_plot(lambda: tree.shap_explain_row_plot(sample, row_index=row),
                      os.path.join(plotdir, f'{tag}_shap_row_insolvent.png'))
        top_cols = []
        for v in vi["variable"]:
            col = v if v in x else v.split('.')[0]
            if col in x and col not in top_cols:
                top_cols.append(col)
            if len(top_cols) == N_PDP:
                break
        for col in top_cols:
            save_plot(lambda: tree.pd_plot(sample, column=col), os.path.join(plotdir, f'{tag}_pdp_{col}.png'))

    try:
        h2o.cluster().shutdown(prompt=False)
    except Exception:
        pass

    subprocess.run(['pkill', '-9', '-f', 'h2o.jar'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(20)
    print(f"=== Finished leak-free {VERSION} run for horizon h{h} ===\n")


def main():
    for h in HORIZONS:
        run_horizon(h)


if __name__ == '__main__':
    main()
