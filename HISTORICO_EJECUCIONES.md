# Resultados de las ejecuciones AutoML: predicción de insolvencia

_Documento generado el 2026-11-02. Recoge las tres familias de ejecuciones del proyecto, los rankings de algoritmos según el conjunto de variables empleado, la importancia de variables y el inventario de gráficos._

## 1. Resumen

| Ejecución | Fecha | Datos | ¿Válida para el paper? | Motivo |
|---|---|---|---|---|
| **Original 2023** | jun. 2023 | `datasets/10_to_predict_{h}_balanced.csv`, `split_frame` aleatorio | **No** | Balanceo e imputación sobre el dataset completo, split por filas y confusión con el periodo (véase §6). |
| **Leak-free v1** | 2026-11-02 | `data/final/*_leakfree_h{h}.csv` (derivado de los CSV balanceados originales con split por CIF) | **No** | Hereda el train original: la variable de periodo (signo de F43, N14) separa las clases (AUC ≈ 0,99). |
| **Leak-free v2** | 2026-11-02 | `data/final/*_leakfree_v2_h{h}.csv`, reconstruido desde `raw_data.csv` | **Aún no** | Ya no hay fuga temporal ni sobreajuste (CV ≈ test), pero quedan dos artefactos de origen: la ausencia de valor en N11 y la edad N1 (véase §6). |
| **Leak-free v3** | 2026-11-02 | `data/final/*_leakfree_v3_h{h}.csv`: mismas ventanas y split que v2, sin N1 ni N11 | **Candidata** | Sin fuga temporal ni artefactos de N1/N11. AUC en test de 0,943/0,921/0,916; queda por revisar N15 (véase §11.6). |

Conclusión principal: en las tres ejecuciones el algoritmo que lidera el ranking es el mismo (**StackedEnsemble > XGBoost/GBM > DRF/XRT > GLM**). Las métricas absolutas, en cambio, no son comparables: las de 2023 y v1 están infladas por fugas de información.

## 2. Configuración de cada ejecución

| | Original 2023 | Leak-free v1 | Leak-free v2 |
|---|---|---|---|
| Script / notebook | `H10h{1,3,5}.ipynb` | `run_leakfree_python.py` | `run_leakfree_python.py` + `scripts/01_build_windows_v2.py` + `scripts/02_split_preprocess_v2.py` |
| H2O | (versión de 2023) | 3.40.0.4, Java 17 | 3.40.0.4, Java 17 |
| `max_models` | 20 | 10 | 10 |
| Validación cruzada | `nfolds=5` (aleatoria) | `nfolds=3` (aleatoria; 1 fila por CIF) | `fold_column` por CIF (3 folds) |
| `sort_metric` | `auc_pr` | `AUCPR` | `AUCPR` |
| Algoritmos excluidos | ninguno (incluye DeepLearning) | DeepLearning | DeepLearning |
| `max_runtime_secs` | — | 1800 | 1800 |
| Recursos | — | 2 hilos, 4 GB | 2 hilos, 4 GB |
| Evaluación en test | `aml.explain(test)` sobre un test procedente de `split_frame` | ninguna | sí, test sin balancear y con la prevalencia real |
| Resultados | `H10h{h}.csv`, `H10h{h}_pf.csv`, `H10h{h}/` | `runs/leakfree_h{h}/` | `runs/leakfree_v2_h{h}/` (incluye `*_test_metrics.csv`) |

### Tamaño de los datos

| | h1 | h3 | h5 |
|---|---|---|---|
| Train v1 (balanceado) | 12.286 | 16.420 | 10.850 |
| Test v1 (no usado) | 12.423 (8,6 % positivos) | 10.210 (7,0 %) | 5.164 (8,6 %) |
| Ventanas v2 (sin balancear) | 487.110 | 376.635 | 278.464 |
| Train v2 balanceado | 5.070 | 12.410 | 16.230 |
| Test v2 (prevalencia real) | 146.412 (0,74 %) | 112.905 (2,35 %) | 83.862 (4,10 %) |

## 3. Variables empleadas

Cada variable base aparece con 10 retardos: `_1` = año t (el más reciente) … `_10` = año t−9. Las variables estáticas (N3, N4, N5, N8, N15) se toman del último año.

| Conjunto | Columnas | Variables base | Variables |
|---|---|---|---|
| Original 2023 | 327 | 37 | F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F27, F28, F29, F30, F31, F33, F34, F35, F36, F37, F38, F39, F40, F41, F42, F43, N2, N3, N4, N5, N6, N7, N8, N12, N13, N14, N15 |
| Leak-free v1 | 327 | 37 | F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F27, F28, F29, F30, F31, F33, F34, F35, F36, F37, F38, F39, F40, F41, F42, F43, N2, N3, N4, N5, N6, N7, N8, N12, N13, N14, N15 |
| Leak-free v2 | 371 | 41 | F16, F17, F18, F19, F20, F21, F22, F23, F24, F25, F27, F28, F29, F30, F31, F33, F34, F35, F36, F37, F38, F39, F40, F41, F42, F43, N1, N2, N3, N4, N5, N6, N7, N8, N9, N10, N11, N12, N13, N14, N15 |

**Diferencias entre conjuntos:**
- **Original y v1** usan las mismas 37 variables base. N1 (antigüedad), N9, N10, N11 (sexo del gerente), F26, F32 y FCA se eliminaron analizando el dataset completo.
- **v2** decide qué eliminar solo con train (umbral: más del 50 % de valores ausentes). Elimina F26 y F32 (81–83 % ausentes) y FCA (fechas posteriores al año observado; 100 % ausente en las solventes). Recupera N1, N9, N10 y N11.
- **v2** corrige además el signo de F27, F30, F42 y F43 en los años anteriores a 2008 (cambio del PGC 2007).

## 4. Ranking de algoritmos por familia

Mejor modelo de cada familia. Para 2023, v1 y v2 (CV) la métrica es el AUCPR en validación cruzada sobre train balanceado; en la última tabla es el AUCPR en test, con la prevalencia real.

**Original 2023**
| Algoritmo       |   Rank |       h1 |     h3 |       h5 |   Media |
|:----------------|-------:|---------:|-------:|---------:|--------:|
| StackedEnsemble |      1 |   0.9931 | 0.9886 |   0.9830 |  0.9882 |
| XGBoost         |      2 |   0.9917 | 0.9876 |   0.9820 |  0.9871 |
| GBM             |      3 |   0.9904 | 0.9820 |   0.9728 |  0.9817 |
| DRF             |      4 |   0.9745 | 0.9646 |   0.9599 |  0.9664 |
| XRT             |      5 |   0.9744 | 0.9666 |   0.9560 |  0.9657 |
| DeepLearning    |      6 |   0.9268 | 0.9163 |   0.9267 |  0.9233 |
| GLM             |      7 | —      | 0.8777 | —      |  0.8777 |
**Leak-free v1**
| Algoritmo       |   Rank |     h1 |     h3 |     h5 |   Media |
|:----------------|-------:|-------:|-------:|-------:|--------:|
| StackedEnsemble |      1 | 0.9928 | 0.9912 | 0.9868 |  0.9903 |
| XGBoost         |      2 | 0.9919 | 0.9903 | 0.9851 |  0.9891 |
| GBM             |      3 | 0.9901 | 0.9864 | 0.9817 |  0.9861 |
| DRF             |      4 | 0.9779 | 0.9743 | 0.9724 |  0.9749 |
| XRT             |      5 | 0.9780 | 0.9728 | 0.9696 |  0.9735 |
| GLM             |      6 | 0.8862 | 0.9098 | 0.8285 |  0.8749 |
**Leak-free v2 (CV)**
| Algoritmo       |   Rank |     h1 |     h3 |     h5 |   Media |
|:----------------|-------:|-------:|-------:|-------:|--------:|
| StackedEnsemble |      1 | 0.9609 | 0.9478 | 0.9534 |  0.9540 |
| GBM             |      2 | 0.9609 | 0.9434 | 0.9498 |  0.9514 |
| XGBoost         |      3 | 0.9579 | 0.9391 | 0.9476 |  0.9482 |
| XRT             |      4 | 0.9222 | 0.9146 | 0.9286 |  0.9218 |
| DRF             |      5 | 0.9208 | 0.9097 | 0.9242 |  0.9182 |
| GLM             |      6 | 0.8006 | 0.8659 | 0.8573 |  0.8412 |
**Leak-free v2 (test)**
| Algoritmo       |   Rank |     h1 |     h3 |     h5 |   Media |
|:----------------|-------:|-------:|-------:|-------:|--------:|
| StackedEnsemble |      1 | 0.2466 | 0.4264 | 0.5806 |  0.4179 |
| GBM             |      2 | 0.2345 | 0.3902 | 0.5433 |  0.3893 |
| XGBoost         |      3 | 0.2250 | 0.3835 | 0.5513 |  0.3866 |
| XRT             |      4 | 0.1136 | 0.2602 | 0.4198 |  0.2645 |
| DRF             |      5 | 0.1072 | 0.2462 | 0.3988 |  0.2508 |
| GLM             |      6 | 0.0404 | 0.1713 | 0.3219 |  0.1779 |

Observaciones:
- El orden de las familias es estable en las tres ejecuciones. Lo único que cambia en v2 es que GBM supera a XGBoost.
- En v2 el AUC en test (≈ 0,96) coincide con el de CV, así que no hay sobreajuste. El AUCPR en test es mucho menor porque la prevalencia es del 0,74–4,1 %. Para h1, un AUCPR de 0,247 equivale a unas 33 veces la tasa base.
- GLM queda siempre último. En 2023 no aparece en los leaderboards de h1 y h5 que se conservaron.

## 5. Leaderboards completos

### 5.1 Original 2023: leaderboard completo (`H10h{h}.csv`, el que usa `pareto_front.ipynb`)

> ⚠️ Estos CSV vienen de **ejecuciones distintas** de las que generaron los modelos guardados y los notebooks. Por ejemplo, para h1 `H10h1.csv` corresponde a `AutoML_1_20230622_152146`, mientras que `H10h1_pf.csv`, `H10h1/` y el notebook corresponden a `AutoML_1_20230623_111421`. Lo mismo ocurre en h3 y h5 (h5 incluso es del 14-jun).

#### h1 (n=20)
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|----------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9931 |     0.9937 |    0.0974 |             11.4000 |                 0.0451 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9923 |     0.9930 |    0.1017 |              6.8000 |                 0.0200 |
|   3 | XGBoost_3                      | XGBoost         |       0.9917 |     0.9924 |    0.1056 |              4.7000 |                 0.0044 |
|   4 | XGBoost_grid_1_model_1         | XGBoost         |       0.9912 |     0.9918 |    0.1158 |              4.3000 |                 0.0064 |
|   5 | XGBoost_grid_1_model_3         | XGBoost         |       0.9910 |     0.9920 |    0.1069 |              6.3000 |                 0.0065 |
|   6 | XGBoost_grid_1_model_2         | XGBoost         |       0.9907 |     0.9919 |    0.1083 |              3.4000 |                 0.0056 |
|   7 | XGBoost_1                      | XGBoost         |       0.9905 |     0.9916 |    0.1115 |              3.7000 |                 0.0058 |
|   8 | GBM_1                          | GBM             |       0.9904 |     0.9911 |    0.1167 |             25.3000 |                 0.0174 |
|   9 | GBM_grid_1_model_2             | GBM             |       0.9901 |     0.9909 |    0.1195 |             15.9000 |                 0.0146 |
|  10 | XGBoost_2                      | XGBoost         |       0.9898 |     0.9913 |    0.1151 |              4.1000 |                 0.0073 |
|  11 | GBM_4                          | GBM             |       0.9892 |     0.9899 |    0.1272 |             16.6000 |                 0.0111 |
|  12 | GBM_3                          | GBM             |       0.9891 |     0.9901 |    0.1237 |             14.7000 |                 0.0122 |
|  13 | GBM_2                          | GBM             |       0.9887 |     0.9898 |    0.1250 |             12.8000 |                 0.0159 |
|  14 | GBM_5                          | GBM             |       0.9882 |     0.9894 |    0.1279 |             16.5000 |                 0.0133 |
|  15 | GBM_grid_1_model_1             | GBM             |       0.9880 |     0.9888 |    0.1341 |             18.5000 |                 0.0131 |
|  16 | DRF_1                          | DRF             |       0.9745 |     0.9771 |    0.2529 |             12.8000 |                 0.0120 |
|  17 | XRT_1                          | DRF             |       0.9744 |     0.9774 |    0.2589 |             12.2000 |                 0.0098 |
|  18 | DeepLearning_1                 | DeepLearning    |       0.9268 |     0.9406 |    0.3604 |              2.6000 |                 0.0074 |
|  19 | DeepLearning_grid_3_model_1    | DeepLearning    |       0.9124 |     0.9179 |    1.4118 |             75.6000 |                 0.0273 |
|  20 | DeepLearning_grid_2_model_1    | DeepLearning    |       0.8995 |     0.9272 |    0.8656 |             76.4000 |                 0.0155 |
#### h3 (n=20)
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|----------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9886 |     0.9903 |    0.1160 |             21.1000 |                 0.0291 |
|   2 | XGBoost_grid_1_model_2         | XGBoost         |       0.9876 |     0.9891 |    0.1239 |              4.5000 |                 0.0053 |
|   3 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9869 |     0.9890 |    0.1217 |             13.8000 |                 0.0218 |
|   4 | XGBoost_grid_1_model_1         | XGBoost         |       0.9866 |     0.9887 |    0.1335 |             10.8000 |                 0.0091 |
|   5 | XGBoost_grid_1_model_3         | XGBoost         |       0.9861 |     0.9887 |    0.1269 |             10.7000 |                 0.0067 |
|   6 | XGBoost_3                      | XGBoost         |       0.9861 |     0.9885 |    0.1267 |              7.1000 |                 0.0079 |
|   7 | XGBoost_2                      | XGBoost         |       0.9857 |     0.9880 |    0.1392 |              5.7000 |                 0.0066 |
|   8 | XGBoost_1                      | XGBoost         |       0.9848 |     0.9879 |    0.1309 |              7.8000 |                 0.0071 |
|   9 | GBM_grid_1_model_2             | GBM             |       0.9820 |     0.9844 |    0.1521 |             37.7000 |                 0.0182 |
|  10 | GBM_4                          | GBM             |       0.9815 |     0.9842 |    0.1564 |             21.0000 |                 0.0179 |
|  11 | GBM_2                          | GBM             |       0.9814 |     0.9841 |    0.1544 |             18.0000 |                 0.0171 |
|  12 | GBM_5                          | GBM             |       0.9811 |     0.9836 |    0.1562 |             16.8000 |                 0.0196 |
|  13 | GBM_3                          | GBM             |       0.9809 |     0.9840 |    0.1546 |             16.4000 |                 0.0182 |
|  14 | GBM_1                          | GBM             |       0.9808 |     0.9838 |    0.1543 |             22.2000 |                 0.0192 |
|  15 | GBM_grid_1_model_1             | GBM             |       0.9776 |     0.9816 |    0.1697 |             51.8000 |                 0.0169 |
|  16 | XRT_1                          | DRF             |       0.9666 |     0.9704 |    0.2676 |             14.0000 |                 0.0141 |
|  17 | DRF_1                          | DRF             |       0.9646 |     0.9696 |    0.2646 |             13.3000 |                 0.0140 |
|  18 | DeepLearning_1                 | DeepLearning    |       0.9163 |     0.9357 |    1.0199 |              3.0000 |                 0.0089 |
|  19 | DeepLearning_grid_1_model_1    | DeepLearning    |       0.9027 |     0.9286 |    0.7408 |             62.1000 |                 0.0183 |
|  20 | GLM_1                          | GLM             |       0.8777 |     0.9098 |    0.4452 |             11.8000 |                 0.0037 |
#### h5 (n=20)
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|----------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9830 |     0.9855 |    0.1433 |             11.1000 |                 0.0353 |
|   2 | XGBoost_3                      | XGBoost         |       0.9820 |     0.9849 |    0.1494 |              3.2000 |                 0.0051 |
|   3 | XGBoost_grid_1_model_3         | XGBoost         |       0.9807 |     0.9838 |    0.1564 |              4.5000 |                 0.0067 |
|   4 | XGBoost_2                      | XGBoost         |       0.9789 |     0.9827 |    0.1627 |              4.3000 |                 0.0051 |
|   5 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9788 |     0.9826 |    0.1565 |              5.9000 |                 0.0172 |
|   6 | XGBoost_grid_1_model_2         | XGBoost         |       0.9786 |     0.9824 |    0.1579 |              3.0000 |                 0.0050 |
|   7 | XGBoost_1                      | XGBoost         |       0.9773 |     0.9814 |    0.1654 |              3.8000 |                 0.0056 |
|   8 | XGBoost_grid_1_model_1         | XGBoost         |       0.9766 |     0.9818 |    0.1654 |              3.2000 |                 0.0052 |
|   9 | GBM_4                          | GBM             |       0.9728 |     0.9776 |    0.1863 |             11.6000 |                 0.0132 |
|  10 | GBM_grid_1_model_2             | GBM             |       0.9723 |     0.9776 |    0.1827 |              9.6000 |                 0.0107 |
|  11 | GBM_5                          | GBM             |       0.9721 |     0.9766 |    0.1888 |              8.1000 |                 0.0107 |
|  12 | GBM_2                          | GBM             |       0.9718 |     0.9763 |    0.1891 |              9.4000 |                 0.0103 |
|  13 | GBM_3                          | GBM             |       0.9709 |     0.9767 |    0.1871 |             10.5000 |                 0.0108 |
|  14 | GBM_1                          | GBM             |       0.9704 |     0.9764 |    0.1887 |             19.3000 |                 0.0142 |
|  15 | GBM_grid_1_model_1             | GBM             |       0.9671 |     0.9746 |    0.1969 |             11.2000 |                 0.0100 |
|  16 | DRF_1                          | DRF             |       0.9599 |     0.9664 |    0.2679 |              9.5000 |                 0.0114 |
|  17 | XRT_1                          | DRF             |       0.9560 |     0.9643 |    0.2780 |              9.7000 |                 0.0108 |
|  18 | DeepLearning_1                 | DeepLearning    |       0.9267 |     0.9443 |    0.5853 |              2.2000 |                 0.0073 |
|  19 | DeepLearning_grid_3_model_1    | DeepLearning    |       0.9168 |     0.9244 |    1.1296 |             63.2000 |                 0.0211 |
|  20 | DeepLearning_grid_2_model_1    | DeepLearning    |       0.8376 |     0.8977 |    1.7112 |             77.2000 |                 0.0175 |

### 5.2 Original 2023: frente de Pareto exportado (`H10h{h}_pf.csv`)

#### h1 (n=6)
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|----------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9910 |     0.9919 |    0.1093 |             11.3000 |                 0.0367 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9907 |     0.9915 |    0.1125 |              6.4000 |                 0.0167 |
|   3 | XGBoost_grid_1_model_3         | XGBoost         |       0.9894 |     0.9907 |    0.1190 |              5.0000 |                 0.0040 |
|   4 | XGBoost_3                      | XGBoost         |       0.9892 |     0.9907 |    0.1202 |              4.0000 |                 0.0036 |
|   5 | XGBoost_grid_1_model_2         | XGBoost         |       0.9883 |     0.9901 |    0.1188 |              3.2000 |                 0.0031 |
|   6 | GLM_1                          | GLM             |       0.8846 |     0.9114 |    0.4350 |              6.3000 |                 0.0030 |
#### h3 (n=5)
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|----------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9904 |     0.9913 |    0.1132 |             30.5000 |                 0.0707 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9893 |     0.9904 |    0.1190 |             14.6000 |                 0.0378 |
|   3 | XGBoost_3                      | XGBoost         |       0.9889 |     0.9901 |    0.1229 |              4.6000 |                 0.0064 |
|   4 | XGBoost_grid_1_model_1         | XGBoost         |       0.9881 |     0.9896 |    0.1275 |              5.0000 |                 0.0059 |
|   5 | GLM_1                          | GLM             |       0.8743 |     0.9121 |    0.4420 |             11.4000 |                 0.0035 |
#### h5 (n=6)
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|----------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9828 |     0.9856 |    0.1417 |              7.6000 |                 0.0201 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9808 |     0.9842 |    0.1485 |              4.4000 |                 0.0102 |
|   3 | XGBoost_grid_1_model_2         | XGBoost         |       0.9804 |     0.9840 |    0.1492 |              2.7000 |                 0.0041 |
|   4 | XGBoost_3                      | XGBoost         |       0.9801 |     0.9836 |    0.1543 |              2.6000 |                 0.0027 |
|   5 | XGBoost_1                      | XGBoost         |       0.9792 |     0.9829 |    0.1554 |             94.9000 |                 0.0026 |
|   6 | GLM_1                          | GLM             |       0.8386 |     0.8808 |    0.5434 |              0.1000 |                 0.0014 |

### 5.3 Leak-free v1 (`runs/leakfree_h{h}/`)

El tiempo de entrenamiento se ha obtenido de los modelos guardados. No hay tiempo de predicción por fila (véase §7).

#### h1
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss (CV) |   Entrenamiento (s) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|---------------:|--------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9928 |     0.9920 |         0.1095 |             10.8000 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9924 |     0.9914 |         0.1129 |              9.3000 |
|   3 | XGBoost_3                      | XGBoost         |       0.9919 |     0.9905 |         0.1216 |              6.8000 |
|   4 | XGBoost_1                      | XGBoost         |       0.9913 |     0.9901 |         0.1233 |              9.0000 |
|   5 | XGBoost_2                      | XGBoost         |       0.9911 |     0.9900 |         0.1266 |              9.6000 |
|   6 | GBM_1                          | GBM             |       0.9901 |     0.9892 |         0.1293 |             38.6000 |
|   7 | GBM_2                          | GBM             |       0.9900 |     0.9888 |         0.1338 |             24.5000 |
|   8 | GBM_3                          | GBM             |       0.9889 |     0.9880 |         0.1383 |             27.1000 |
|   9 | GBM_4                          | GBM             |       0.9888 |     0.9876 |         0.1417 |             30.3000 |
|  10 | XRT_1                          | XRT             |       0.9780 |     0.9754 |         0.2687 |             22.4000 |
|  11 | DRF_1                          | DRF             |       0.9779 |     0.9753 |         0.2631 |             22.5000 |
|  12 | GLM_1                          | GLM             |       0.8862 |     0.9042 |         0.4559 |             13.6000 |
#### h3
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss (CV) |   Entrenamiento (s) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|---------------:|--------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9912 |     0.9896 |         0.1223 |             12.7000 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9906 |     0.9890 |         0.1264 |             10.7000 |
|   3 | XGBoost_3                      | XGBoost         |       0.9903 |     0.9885 |         0.1315 |              9.8000 |
|   4 | XGBoost_1                      | XGBoost         |       0.9894 |     0.9872 |         0.1356 |             12.0000 |
|   5 | XGBoost_2                      | XGBoost         |       0.9893 |     0.9872 |         0.1393 |             10.7000 |
|   6 | GBM_3                          | GBM             |       0.9864 |     0.9839 |         0.1568 |             34.2000 |
|   7 | GBM_1                          | GBM             |       0.9862 |     0.9836 |         0.1554 |             50.4000 |
|   8 | GBM_2                          | GBM             |       0.9862 |     0.9837 |         0.1571 |             25.6000 |
|   9 | GBM_4                          | GBM             |       0.9858 |     0.9833 |         0.1629 |             52.7000 |
|  10 | DRF_1                          | DRF             |       0.9743 |     0.9692 |         0.2648 |             23.5000 |
|  11 | XRT_1                          | XRT             |       0.9728 |     0.9669 |         0.2722 |             22.5000 |
|  12 | GLM_1                          | GLM             |       0.9098 |     0.9154 |         0.4215 |              3.4000 |
#### h5
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss (CV) |   Entrenamiento (s) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|---------------:|--------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9868 |     0.9841 |         0.1495 |              8.6000 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9858 |     0.9831 |         0.1543 |              8.1000 |
|   3 | XGBoost_1                      | XGBoost         |       0.9851 |     0.9824 |         0.1601 |              6.9000 |
|   4 | XGBoost_3                      | XGBoost         |       0.9850 |     0.9815 |         0.1634 |              5.4000 |
|   5 | XGBoost_2                      | XGBoost         |       0.9848 |     0.9815 |         0.1675 |              6.8000 |
|   6 | GBM_2                          | GBM             |       0.9817 |     0.9780 |         0.1796 |             15.9000 |
|   7 | GBM_1                          | GBM             |       0.9816 |     0.9783 |         0.1790 |             30.0000 |
|   8 | GBM_3                          | GBM             |       0.9812 |     0.9775 |         0.1828 |             17.4000 |
|   9 | GBM_4                          | GBM             |       0.9808 |     0.9771 |         0.1858 |             18.0000 |
|  10 | DRF_1                          | DRF             |       0.9724 |     0.9672 |         0.2689 |             16.4000 |
|  11 | XRT_1                          | XRT             |       0.9696 |     0.9646 |         0.2766 |             20.3000 |
|  12 | GLM_1                          | GLM             |       0.8285 |     0.8444 |         0.5973 |              5.6000 |

### 5.4 Leak-free v2 (`runs/leakfree_v2_h{h}/`)

#### h1
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss (CV) |   AUC (test) |   AUCPR (test) |   Entrenamiento (s) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|---------------:|-------------:|---------------:|--------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9609 |     0.9668 |         0.2304 |       0.9625 |         0.2466 |              6.2000 |
|   2 | GBM_4                          | GBM             |       0.9609 |     0.9649 |         0.2396 |       0.9572 |         0.2094 |             17.1000 |
|   3 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9599 |     0.9662 |         0.2327 |       0.9611 |         0.2381 |              5.9000 |
|   4 | GBM_3                          | GBM             |       0.9586 |     0.9635 |         0.2425 |       0.9585 |         0.2300 |             17.1000 |
|   5 | GBM_2                          | GBM             |       0.9585 |     0.9632 |         0.2446 |       0.9583 |         0.2328 |             12.3000 |
|   6 | GBM_1                          | GBM             |       0.9581 |     0.9626 |         0.2450 |       0.9579 |         0.2345 |             18.6000 |
|   7 | XGBoost_3                      | XGBoost         |       0.9579 |     0.9615 |         0.2567 |       0.9581 |         0.2250 |              4.2000 |
|   8 | XGBoost_2                      | XGBoost         |       0.9539 |     0.9606 |         0.2625 |       0.9552 |         0.1847 |              6.8000 |
|   9 | XGBoost_1                      | XGBoost         |       0.9519 |     0.9589 |         0.2600 |       0.9562 |         0.1950 |              6.7000 |
|  10 | XRT_1                          | XRT             |       0.9222 |     0.9387 |         0.3376 |       0.9337 |         0.1136 |             13.2000 |
|  11 | DRF_1                          | DRF             |       0.9208 |     0.9373 |         0.3451 |       0.9301 |         0.1072 |             12.0000 |
|  12 | GLM_1                          | GLM             |       0.8006 |     0.8316 |         0.6689 |       0.8366 |         0.0404 |              2.3000 |
#### h3
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss (CV) |   AUC (test) |   AUCPR (test) |   Entrenamiento (s) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|---------------:|-------------:|---------------:|--------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9478 |     0.9559 |         0.2625 |       0.9590 |         0.4264 |              9.0000 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9462 |     0.9545 |         0.2668 |       0.9571 |         0.4103 |              8.0000 |
|   3 | GBM_4                          | GBM             |       0.9434 |     0.9517 |         0.2793 |       0.9534 |         0.3831 |             28.9000 |
|   4 | GBM_1                          | GBM             |       0.9433 |     0.9515 |         0.2768 |       0.9539 |         0.3870 |             35.2000 |
|   5 | GBM_3                          | GBM             |       0.9422 |     0.9508 |         0.2800 |       0.9535 |         0.3902 |             19.5000 |
|   6 | GBM_2                          | GBM             |       0.9415 |     0.9506 |         0.2798 |       0.9527 |         0.3810 |             15.9000 |
|   7 | XGBoost_3                      | XGBoost         |       0.9391 |     0.9493 |         0.2849 |       0.9528 |         0.3835 |             11.5000 |
|   8 | XGBoost_2                      | XGBoost         |       0.9369 |     0.9466 |         0.3045 |       0.9490 |         0.3459 |             16.2000 |
|   9 | XGBoost_1                      | XGBoost         |       0.9355 |     0.9482 |         0.2933 |       0.9518 |         0.3597 |             17.3000 |
|  10 | XRT_1                          | XRT             |       0.9146 |     0.9298 |         0.3465 |       0.9328 |         0.2602 |             22.4000 |
|  11 | DRF_1                          | DRF             |       0.9097 |     0.9248 |         0.3553 |       0.9261 |         0.2462 |             21.8000 |
|  12 | GLM_1                          | GLM             |       0.8659 |     0.8834 |         0.4726 |       0.8895 |         0.1713 |             18.5000 |
#### h5
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   Logloss (CV) |   AUC (test) |   AUCPR (test) |   Entrenamiento (s) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|---------------:|-------------:|---------------:|--------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9534 |     0.9611 |         0.2476 |       0.9617 |         0.5806 |             12.3000 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9526 |     0.9598 |         0.2526 |       0.9596 |         0.5628 |             10.2000 |
|   3 | GBM_4                          | GBM             |       0.9498 |     0.9565 |         0.2662 |       0.9541 |         0.5238 |             50.4000 |
|   4 | GBM_3                          | GBM             |       0.9493 |     0.9563 |         0.2653 |       0.9551 |         0.5332 |             22.3000 |
|   5 | GBM_1                          | GBM             |       0.9489 |     0.9572 |         0.2609 |       0.9555 |         0.5382 |             34.1000 |
|   6 | XGBoost_3                      | XGBoost         |       0.9476 |     0.9562 |         0.2661 |       0.9582 |         0.5513 |             13.7000 |
|   7 | GBM_2                          | GBM             |       0.9470 |     0.9562 |         0.2640 |       0.9555 |         0.5433 |             22.6000 |
|   8 | XGBoost_2                      | XGBoost         |       0.9428 |     0.9530 |         0.2860 |       0.9551 |         0.5199 |             32.1000 |
|   9 | XGBoost_1                      | XGBoost         |       0.9412 |     0.9529 |         0.2805 |       0.9563 |         0.5203 |             21.4000 |
|  10 | XRT_1                          | XRT             |       0.9286 |     0.9388 |         0.3295 |       0.9375 |         0.4198 |             25.7000 |
|  11 | DRF_1                          | DRF             |       0.9242 |     0.9350 |         0.3337 |       0.9322 |         0.3988 |             23.0000 |
|  12 | GLM_1                          | GLM             |       0.8573 |     0.8664 |         0.4927 |       0.8882 |         0.3219 |              2.6000 |

## 6. Importancia de variables

Top 10 por modelo (% de la importancia total), calculado sobre los modelos guardados.

- v1 h1 GBM_1: N14_10 (54.0%), F22_1 (9.3%), F28_3 (7.3%), N14_9 (2.1%), F30_1 (1.4%), F28_2 (1.3%), F31_1 (1.3%), N14_3 (1.2%), F21_1 (1.2%), F28_1 (1.2%)
- v1 h1 XGBoost_1: F43_10 (41.5%), F30_1 (13.1%), F43_9 (7.0%), F20_1 (1.9%), F27_1 (1.8%), F43_8 (1.6%), F41_1 (1.5%), F28_2 (1.3%), F22_1 (1.3%), F25_1 (1.2%)
- v1 h3 GBM_1: N14_10 (64.6%), N14_9 (5.6%), F28_2 (3.5%), N14_8 (1.8%), F28_1 (1.7%), F20_10 (1.3%), F28_3 (1.1%), N15 (1.0%), N14_3 (0.9%), N14_7 (0.8%)
- v1 h3 XGBoost_1: F43_10 (49.1%), F43_9 (11.2%), F27_9 (1.9%), F27_1 (1.8%), F43_8 (1.8%), N14_10 (1.2%), F28_1 (0.9%), F27_10 (0.9%), F27_8 (0.8%), F28_2 (0.8%)
- v1 h5 GBM_1: N14_10 (71.3%), N14_9 (6.2%), N14_8 (3.1%), F28_1 (1.9%), N15 (1.0%), N14_3 (0.8%), F20_8 (0.7%), N14_4 (0.5%), N14_6 (0.5%), N14_7 (0.4%)
- v1 h5 XGBoost_1: F43_8 (40.8%), N14_10 (18.8%), F27_1 (2.4%), F43_7 (2.4%), F27_7 (1.2%), F43_4 (1.1%), F27_8 (0.9%), N15.No (0.9%), F29_8 (0.8%), F43_6 (0.6%)
- v2 h1 GBM_1: F21_1 (31.8%), F28_1 (10.1%), N1_1 (8.6%), F20_1 (4.9%), F25_1 (2.7%), F22_1 (2.7%), N1_8 (2.2%), F37_2 (2.2%), N1_3 (2.1%), N11_9 (1.7%)
- v2 h1 XGBoost_1: F25_1 (23.2%), N1_1 (12.1%), F30_1 (9.3%), F27_1 (3.8%), F41_1 (3.2%), N11_1.missing(NA) (2.6%), F31_1 (1.7%), F37_1 (1.3%), F22_1 (1.1%), N1_2 (1.1%)
- v2 h3 GBM_1: F28_1 (17.1%), N1_1 (15.7%), N11_7 (5.5%), N11_3 (3.5%), N1_6 (3.3%), N1_10 (2.9%), F21_1 (2.9%), N11_2 (2.3%), N1_9 (2.2%), F28_2 (2.2%)
- v2 h3 XGBoost_1: F27_1 (21.0%), N1_1 (13.5%), F25_1 (3.6%), N11_1.missing(NA) (3.0%), N1_2 (2.5%), N11_3.missing(NA) (2.2%), N11_4.missing(NA) (2.0%), N11_2.missing(NA) (1.6%), F41_1 (1.3%), F36_1 (1.0%)
- v2 h5 GBM_1: N1_1 (16.5%), N1_5 (8.0%), N1_4 (6.8%), F28_1 (6.5%), N1_7 (5.0%), N11_1 (4.4%), N11_8 (4.3%), N11_7 (4.2%), F37_1 (2.6%), N11_6 (2.1%)
- v2 h5 XGBoost_1: N1_1 (22.7%), F27_1 (14.3%), N11_2.missing(NA) (5.2%), N11_1.missing(NA) (4.9%), F29_1 (1.7%), F28_1 (1.7%), F25_1 (1.3%), N2_3 (1.1%), N1_2 (1.0%), N2_1 (0.9%)

**Interpretación:**
- **v1:** dominan variables que codifican el **periodo**, no la salud financiera.
  - `F43_10`/`F43_9` (XGBoost): el signo de F43 cambia en 2008 en todas las empresas. En train, el 96 % de las ventanas solventes son posteriores a 2008, frente al 16 % de las insolventes.
  - `N14_10` (GBM, 54–71 %): días de retraso en presentar las cuentas, que también cambia de nivel entre periodos (mediana de 47 → 74 días).
- **v2, h1:** dominan ratios financieros del último año (F21, F28, F25, F20), lo cual es razonable.
- **v2, h3/h5:** dominan **N1** (antigüedad) y **N11** (sexo del gerente, sobre todo su ausencia). Las dos son artefactos de cómo se construyó `raw_data.csv`:
  - N11 está ausente en el 68 % de las filas de empresas insolventes y en el 8 % de las solventes.
  - El 31 % de las empresas solventes tiene años con datos *anteriores a su fundación* (hasta 19 años, con N1=0), mientras que ninguna insolvente los tiene.

## 7. Gráficos generados

### ¿Qué existe?

| Gráfico | Dónde | Ejecución de origen |
|---|---|---|
| Importancia de variables (`varimp_plot`) | `varimp.png` (27-jun-2023) y celda 15 de `H10h1.ipynb` | Original 2023, solo h1 |
| `aml.explain(test)`: 10 gráficos por horizonte (importancia de variables, heatmap de varimp, correlación entre modelos, SHAP summary, dependencia parcial e ICE) | `H10h{1,3,5}.ipynb` | Original 2023 |
| Curva de aprendizaje del líder | `H10h{1,3,5}.ipynb` | Original 2023 |
| **Pareto AUC frente a tiempo de predicción** (`aml.pareto_front(x_metric="predict_time_per_row_ms", y_metric="auc")`) | `H10h{1,3,5}.ipynb`, celda `pf` | Original 2023 |
| Coeficientes del metalearner (`std_coef_plot`) | `H10h{1,3,5}.ipynb` | Original 2023 |
| SHAP de una fila (`shap_explain_row_plot`) | `H10h1.ipynb` | Original 2023, solo h1 |
| **Pareto combinado h1/h3/h5** (AUC frente a `predict_time_per_row_ms`, coloreado por algoritmo) | `pareto_front.ipynb` | Original 2023 (a partir de `H10h{h}.csv`, que es otra ejecución; véase §5.1) |

Notas:
- `H10h{1,3,5}.ipynb` aparecen como modificados en git porque se volvieron a guardar como JSON de una sola línea, pero conservan sus salidas y gráficos.
- Los notebooks de `notebooks_leakfree/` contienen **las mismas imágenes de 2023** (los hashes son idénticos): se copiaron con las salidas antiguas y **no** se volvieron a ejecutar con los datos leak-free.
- `executed_leakfree/H10h1_leakfree_executed.ipynb` no tiene ningún gráfico: falló con el mismo error de `sort_metric="auc_pr"` ("Aborting AutoML after too many consecutive model failures").

### ¿Qué falta?

- **Para v1 y v2 no se ha generado ningún gráfico**: ni de importancia de variables, ni SHAP, ni de Pareto. `run_leakfree_python.py` solo guarda los leaderboards, los modelos y el MOJO del líder.
- **No se pueden obtener los Pareto frente a velocidad de v1 y v2 con los CSV actuales**: los leaderboards se guardaron sin `training_time_ms` ni `predict_time_per_row_ms`, porque falta `extra_columns="ALL"`. El tiempo de entrenamiento se ha recuperado de los modelos guardados (tablas de §5.3 y §5.4). El tiempo de predicción por fila habría que medirlo de nuevo puntuando el test con cada modelo, o bien relanzar AutoML con `aml.leaderboard` → `h2o.automl.get_leaderboard(aml, extra_columns="ALL")`.

## 8. Leak-free v3: variables y empresas descartadas

La versión v3 (`scripts/01_build_windows_v2.py` y `scripts/02_split_preprocess_v2.py` con `VERSION='v3'`) parte de v2 y elimina dos variables. El criterio es: **se descarta una variable cuando su valor depende de cómo se recogieron los datos de cada tipo de empresa (solvente o insolvente), y no de la situación económica de la empresa.** Si se mantuviera, el modelo aprendería a reconocer de qué fuente viene cada empresa, y no su riesgo de insolvencia. En cambio, **no se descarta ninguna empresa por su desenlace** (véase §8.3).

### 8.1 N1 (antigüedad): descartada

| Evidencia (`raw_data.csv`) | Solventes | Insolventes |
|---|---|---|
| Empresas en las que N1 **retrocede** (vuelve a 0) | 28,9 % | 0 % |
| Empresas con N1=0 en su primer año y primer año = 1999 | 27,9 % | 2 % |
| Empresas con años de datos **anteriores a su fundación** (Year − N1) | 31,2 % | 0 % |
| Filas afectadas | 174.575 (10,9 % del total) | 0 |

- En casi un tercio de las solventes, N1 **cuenta años desde el inicio del panel** (0 en 1999) y se reinicia en la fecha de fundación. Ejemplo: N1 = 0,1,…,10 entre 1999 y 2009 y luego N1 = 0 en 2010. En el 92,7 % de las empresas con años "pre-fundación" se da este patrón de reinicio.
- En las insolventes N1 es siempre coherente (crece 1 por año).
- Consecuencia: N1 distingue la fuente de los datos, no la edad. En v2, N1 acumulaba el 16–40 % de la importancia en h3/h5, con una mediana de 12 años en las solventes frente a 21 en las insolventes. Esa diferencia se debe al reinicio, no a la edad real.
- Las **filas** pre-fundación se **mantienen**: sus variables financieras varían año a año y tienen la misma tasa de ausencia que el resto (8,2 % frente a 9,5 %), así que parecen datos reales. Lo que no es fiable es solo el contador de edad. Eliminarlas sería además una decisión asimétrica, porque solo afectan a solventes.

### 8.2 N11 (sexo del gerente): descartada

| Evidencia (`raw_data.csv`, años ≥ 2008) | Solventes | Insolventes |
|---|---|---|
| Filas con N11 ausente | 7,9 % | 67,6 % |

- Es la mayor diferencia de ausencia entre tipos de todas las variables (sin contar FCA, que ya se descartó en v2 por la misma razón: 100 % frente a 0 %). Lo probable es que la base de datos no conserve los datos del gerente de las sociedades disueltas.
- En v2, el modelo usaba directamente el indicador `N11_*.missing(NA)` (XGBoost h3/h5) y N11 acumulaba el 7–19 % de la importancia. En la práctica, "no sabemos quién es el gerente" equivalía a "empresa insolvente".
- La variable no se puede corregir: imputarla no elimina la información que lleva el patrón de ausencia.

### 8.3 Empresas "Insolvent" sin año de insolvencia: se mantienen (decisión revisada)

- `raw_data.csv` tiene **58.524** empresas con Type=Insolvent, pero solo **7.961** tienen una fila con Y=1, que marca el año de la insolvencia. Las otras **50.563** tienen Y=0 en todos sus años observados, y el 98 % deja de presentar cuentas antes de 2019. Son empresas que acabaron siendo insolventes, pero el año de su insolvencia no está registrado.
- **Se planteó excluirlas y se descartó.** Excluirlas implica usar la columna Type, que es el desenlace final de la empresa, para decidir qué empresas entran en el estudio. Es una **selección basada en el resultado**, de la misma familia que el *data leakage*:
  - Los negativos quedarían formados solo por empresas que nunca quiebran, y desaparecerían las empresas frágiles que acaban quebrando más tarde, que son las más difíciles de clasificar.
  - Las métricas saldrían optimistas, y también en test, porque el test se filtraría igual.
  - En producción esa exclusión es imposible: al predecir no se sabe qué empresas acabarán quebrando.
- **Por qué es aceptable mantenerlas como negativas:**
  - La regla de etiquetado ya exige que la empresa se observe con Y=0 hasta t+h. Si la insolvencia ocurre después de su último año observado, la etiqueta Y=0 de la ventana es correcta.
  - El posible ruido (insolvencias no registradas dentro del horizonte) hace las métricas **más pesimistas**, nunca más optimistas.
  - Estas empresas proceden de la misma fuente que las insolventes con fecha, lo que impide que el modelo aprenda "fuente de insolventes = positivo".
- Configuración: `EXCLUDE_UNDATED_INSOLVENT = False` en `scripts/01_build_windows_v2.py`. Un intento de ejecución con `True` se detuvo y sus resultados se borraron.

### 8.4 Tamaño de los datos v3

Las ventanas y el split por CIF son los mismos que en v2 (misma semilla). Solo cambian las columnas.

| | h1 | h3 | h5 |
|---|---|---|---|
| Ventanas (sin balancear) | 487.110 | 376.635 | 278.464 |
| Positivos | 3.622 (0,74 %) | 8.856 (2,35 %) | 11.551 (4,15 %) |
| Train balanceado | 5.070 | 12.410 | 16.230 |
| Test (prevalencia real) | 146.412 | 112.905 | 83.862 |
| Variables base | 39 | 39 | 39 |

En v3 se eliminan **F26 y F32** (más del 80 % de ausencia en train), además de FCA, N1 y N11.

## 9. Elementos de análisis: notebooks 2023 frente al script

Revisión celda por celda de `H10h{1,3,5}.ipynb` (versión confirmada en git) y de su equivalente en `run_leakfree_python.py`. Las salidas se guardan en `runs/leakfree_v3_h{h}/`, y los gráficos en `plots/`.

| Elemento en 2023 | Celda | En el script (v3) | Salida |
|---|---|---|---|
| Leaderboard con `training_time_ms`, `predict_time_per_row_ms` y `algo` | `explain` | ✅ `get_leaderboard(aml, extra_columns="ALL")` | `*_leaderboard.csv` |
| Matriz de confusión del líder en test | `explain` | ✅ umbral de F1 máximo | `*_leader_confusion_matrix_test.csv` |
| Curva de aprendizaje del líder | 5 | ✅ | `plots/leader_learning_curve.png` |
| Importancia de variables | `explain`, 15 | ✅ del mejor modelo de árboles (GBM/XGBoost/DRF) | `plots/<modelo>_varimp.png` y `*_varimp.csv` |
| Heatmap de importancia entre modelos | `explain` | ✅ | `plots/varimp_heatmap.png` |
| Correlación entre modelos | `explain` | ✅ sobre una muestra de 10.000 filas de test | `plots/model_correlation.png` |
| SHAP summary | `explain` | ✅ mejor modelo de árboles, muestra de test | `plots/<modelo>_shap_summary.png` |
| PDP de las 5 variables principales | `explain` | ✅ | `plots/<modelo>_pdp_<var>.png` |
| Pareto AUC frente a tiempo de predicción | 7 | ✅ (AUC en CV, igual que en 2023) | `plots/pareto_auc_cv_vs_predict_time.png` y CSV |
| Coeficientes del metalearner y su gráfico | 8–9 | ✅ | `*_metalearner_coef_norm.json` y `plots/metalearner_std_coef.png` |
| Guardar todos los modelos | 10–12 | ✅ y además el MOJO del líder | `runs/leakfree_v3_h{h}/` |
| SHAP de una fila | 16 | ✅ la primera empresa **insolvente** de la muestra (en 2023 era la fila 0, casi siempre solvente) | `plots/<modelo>_shap_row_insolvent.png` |
| **Nuevo:** métricas de todos los modelos en test (AUC, AUCPR, logloss, F1) | — | ✅ | `*_test_metrics.csv` |
| **Nuevo:** Pareto AUCPR (test) frente a tiempo de predicción | — | ✅ | `plots/pareto_aucpr_test_vs_predict_time.png` y CSV |
| **Nuevo:** Pareto AUCPR (test) frente a tiempo de entrenamiento | — | ✅ | `plots/pareto_aucpr_test_vs_training_time.png` y CSV |

Diferencias deliberadas respecto a 2023:
- **SHAP y PDP se calculan sobre el mejor modelo de árboles y no sobre el líder**, porque TreeSHAP no está disponible para StackedEnsemble. El `aml.explain` de 2023 hacía lo mismo internamente.
- Los gráficos de explicación usan una **muestra aleatoria de unas 10.000 filas de test**: el test completo tiene hasta 134.000 filas y H2O tiene 4 GB. Las métricas y los Pareto en test usan **todo** el test.
- Se excluye DeepLearning y se usan 10 modelos en lugar de 20, igual que en v1 y v2.

## 10. Próximos pasos

1. Averiguar cómo se generó `raw_data.csv`: el reinicio de N1, la ausencia de N11 por tipo y las insolventes sin fecha apuntan a que solventes e insolventes se extrajeron por vías distintas.
2. Revisar N14 (días de retraso en presentar las cuentas): cambia de nivel entre periodos (mediana de 47 → 74) y dominaba en v1.
3. Revisar N15 (empresa familiar): su distribución difiere entre los dos grupos de insolventes (véase §11.6).
4. Opcional: una ejecución solo con variables financieras (F*), para medir la aportación de las variables no financieras.
5. Elegir el umbral de decisión según los costes (véase §11.4).


## 11. Resultados de leak-free v3

Ejecución completada el 2026-11-02 (`leakfree_v3_run.log`, sin errores). Resultados en `runs/leakfree_v3_h{1,3,5}/`: 16 gráficos por horizonte.

### 11.1 Efecto de eliminar N1 y N11 (v2 → v3)

Las ventanas, el split y la semilla son los mismos, así que **la diferencia mide solo lo que aportaban N1 y N11**.

| h   | Ejecución        | Líder                       |   AUC CV |   AUC test |   AUCPR test |
|:----|:-----------------|:----------------------------|---------:|-----------:|-------------:|
| h1  | v2 (con N1, N11) | StackedEnsemble_AllModels_1 |   0.9668 |     0.9625 |       0.2466 |
| h1  | v3 (sin N1, N11) | StackedEnsemble_AllModels_1 |   0.9440 |     0.9431 |       0.2306 |
| h3  | v2 (con N1, N11) | StackedEnsemble_AllModels_1 |   0.9559 |     0.9590 |       0.4264 |
| h3  | v3 (sin N1, N11) | StackedEnsemble_AllModels_1 |   0.9162 |     0.9208 |       0.3378 |
| h5  | v2 (con N1, N11) | StackedEnsemble_AllModels_1 |   0.9611 |     0.9617 |       0.5806 |
| h5  | v3 (sin N1, N11) | StackedEnsemble_AllModels_1 |   0.9156 |     0.9155 |       0.4458 |

- El AUC en test baja de forma moderada en h1 (−0,019) y mucho más en h3 y h5 (−0,04). Es justo donde N1 y N11 dominaban la importancia en v2: esa parte del rendimiento venía del artefacto, no de la información financiera.
- Ahora **el rendimiento disminuye con el horizonte** (AUC en test: 0,943 → 0,921 → 0,916), como cabe esperar: predecir a 5 años es más difícil que a 1. En v2, h5 era tan bueno como h1.
- CV y test siguen coincidiendo (diferencia ≤ 0,005), así que no hay sobreajuste.
- El AUCPR crece con h porque crece la tasa de positivos (0,74 % → 4,15 %). Para comparar horizontes es mejor mirar el AUC.

### 11.2 Ranking de familias (AUCPR en test, mejor modelo de cada familia)

| Algoritmo       |   Rank |     h1 |     h3 |     h5 |   Media |
|:----------------|-------:|-------:|-------:|-------:|--------:|
| StackedEnsemble |      1 | 0.2306 | 0.3378 | 0.4458 |  0.3381 |
| XGBoost         |      2 | 0.1847 | 0.2932 | 0.4136 |  0.2972 |
| GBM             |      3 | 0.2100 | 0.2958 | 0.3799 |  0.2952 |
| XRT             |      4 | 0.1299 | 0.1709 | 0.2412 |  0.1807 |
| DRF             |      5 | 0.1322 | 0.1787 | 0.2281 |  0.1796 |
| GLM             |      6 | 0.0232 | 0.0948 | 0.1790 |  0.0990 |

### 11.3 Leaderboards completos

Métricas de CV sobre train balanceado (folds por CIF) y de test sobre el conjunto sin balancear. Los tiempos son los de H2O (`extra_columns="ALL"`).

#### h1
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   AUC (test) |   AUCPR (test) |   F1 (test) |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|-------------:|---------------:|------------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9401 |     0.9440 |       0.9431 |         0.2306 |      0.3129 |              5.0000 |                 0.1085 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9395 |     0.9430 |       0.9403 |         0.2087 |      0.3034 |              4.9000 |                 0.0669 |
|   3 | GBM_2                          | GBM             |       0.9345 |     0.9385 |       0.9355 |         0.1946 |      0.2887 |             10.4000 |                 0.0204 |
|   4 | GBM_1                          | GBM             |       0.9343 |     0.9388 |       0.9374 |         0.2100 |      0.2977 |             15.8000 |                 0.0247 |
|   5 | GBM_4                          | GBM             |       0.9331 |     0.9378 |       0.9339 |         0.1940 |      0.2845 |             14.7000 |                 0.0213 |
|   6 | XGBoost_1                      | XGBoost         |       0.9309 |     0.9345 |       0.9335 |         0.1615 |      0.2568 |              4.4000 |                 0.0120 |
|   7 | GBM_3                          | GBM             |       0.9309 |     0.9374 |       0.9342 |         0.2001 |      0.2928 |             11.7000 |                 0.0188 |
|   8 | XGBoost_3                      | XGBoost         |       0.9298 |     0.9351 |       0.9363 |         0.1847 |      0.2747 |              3.7000 |                 0.0106 |
|   9 | XGBoost_2                      | XGBoost         |       0.9282 |     0.9344 |       0.9329 |         0.1745 |      0.2623 |              5.8000 |                 0.0088 |
|  10 | DRF_1                          | DRF             |       0.9109 |     0.9198 |       0.9185 |         0.1322 |      0.2182 |              9.9000 |                 0.0174 |
|  11 | XRT_1                          | XRT             |       0.9072 |     0.9161 |       0.9124 |         0.1299 |      0.2240 |             15.0000 |                 0.0217 |
|  12 | GLM_1                          | GLM             |       0.7326 |     0.7849 |       0.7893 |         0.0232 |      0.0520 |              2.8000 |                 0.0077 |
#### h3
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   AUC (test) |   AUCPR (test) |   F1 (test) |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|-------------:|---------------:|------------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9084 |     0.9162 |       0.9208 |         0.3378 |      0.3772 |              7.6000 |                 0.0650 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9032 |     0.9122 |       0.9173 |         0.3224 |      0.3668 |              6.7000 |                 0.0379 |
|   3 | XGBoost_3                      | XGBoost         |       0.8962 |     0.9061 |       0.9112 |         0.2932 |      0.3492 |              9.7000 |                 0.0059 |
|   4 | GBM_1                          | GBM             |       0.8951 |     0.9039 |       0.9098 |         0.2958 |      0.3427 |             27.7000 |                 0.0198 |
|   5 | XGBoost_1                      | XGBoost         |       0.8945 |     0.9022 |       0.9056 |         0.2789 |      0.3329 |             17.8000 |                 0.0072 |
|   6 | XGBoost_2                      | XGBoost         |       0.8943 |     0.9032 |       0.9065 |         0.2725 |      0.3315 |             16.6000 |                 0.0074 |
|   7 | GBM_4                          | GBM             |       0.8938 |     0.9019 |       0.9059 |         0.2784 |      0.3334 |             29.5000 |                 0.0177 |
|   8 | GBM_3                          | GBM             |       0.8928 |     0.9019 |       0.9056 |         0.2901 |      0.3353 |             20.6000 |                 0.0164 |
|   9 | GBM_2                          | GBM             |       0.8927 |     0.9014 |       0.9059 |         0.2790 |      0.3350 |             16.6000 |                 0.0154 |
|  10 | DRF_1                          | DRF             |       0.8546 |     0.8710 |       0.8740 |         0.1787 |      0.2536 |             23.7000 |                 0.0180 |
|  11 | XRT_1                          | XRT             |       0.8503 |     0.8658 |       0.8690 |         0.1709 |      0.2448 |             23.0000 |                 0.0198 |
|  12 | GLM_1                          | GLM             |       0.7858 |     0.8166 |       0.8285 |         0.0948 |      0.1817 |             12.1000 |                 0.0067 |
#### h5
|   # | Modelo                         | Algoritmo       |   AUCPR (CV) |   AUC (CV) |   AUC (test) |   AUCPR (test) |   F1 (test) |   Entrenamiento (s) |   Predicción (ms/fila) |
|----:|:-------------------------------|:----------------|-------------:|-----------:|-------------:|---------------:|------------:|--------------------:|-----------------------:|
|   1 | StackedEnsemble_AllModels_1    | StackedEnsemble |       0.9081 |     0.9156 |       0.9155 |         0.4458 |      0.4579 |             10.8000 |                 0.1091 |
|   2 | StackedEnsemble_BestOfFamily_1 | StackedEnsemble |       0.9016 |     0.9103 |       0.9118 |         0.4324 |      0.4457 |              9.1000 |                 0.0346 |
|   3 | XGBoost_3                      | XGBoost         |       0.8946 |     0.9044 |       0.9076 |         0.4136 |      0.4447 |             14.7000 |                 0.0076 |
|   4 | XGBoost_2                      | XGBoost         |       0.8943 |     0.9027 |       0.9004 |         0.3817 |      0.4202 |             22.3000 |                 0.0112 |
|   5 | GBM_1                          | GBM             |       0.8905 |     0.8996 |       0.8991 |         0.3799 |      0.4110 |             43.5000 |                 0.0263 |
|   6 | GBM_2                          | GBM             |       0.8903 |     0.8974 |       0.8992 |         0.3788 |      0.4145 |             19.9000 |                 0.0218 |
|   7 | XGBoost_1                      | XGBoost         |       0.8899 |     0.9000 |       0.9014 |         0.3656 |      0.4123 |             24.5000 |                 0.0096 |
|   8 | GBM_3                          | GBM             |       0.8894 |     0.8974 |       0.8978 |         0.3756 |      0.4060 |             26.1000 |                 0.0244 |
|   9 | GBM_4                          | GBM             |       0.8873 |     0.8960 |       0.8960 |         0.3636 |      0.4021 |             37.7000 |                 0.0263 |
|  10 | XRT_1                          | XRT             |       0.8397 |     0.8546 |       0.8614 |         0.2412 |      0.3130 |             28.9000 |                 0.0310 |
|  11 | DRF_1                          | DRF             |       0.8392 |     0.8550 |       0.8578 |         0.2281 |      0.3102 |             21.6000 |                 0.0290 |
|  12 | GLM_1                          | GLM             |       0.7338 |     0.7697 |       0.8293 |         0.1790 |      0.2828 |              2.8000 |                 0.0035 |

### 11.4 Matriz de confusión del líder en test (umbral de F1 máximo)

| h   |   TP |   FP |   FN |     TN |   Recall |   Precisión |
|:----|-----:|-----:|-----:|-------:|---------:|------------:|
| h1  |  381 |  967 |  706 | 144358 |    0.351 |       0.283 |
| h3  | 1143 | 2267 | 1508 | 107987 |    0.431 |       0.335 |
| h5  | 1609 | 1983 | 1827 |  78443 |    0.468 |       0.448 |

Con el umbral que maximiza F1, el modelo detecta entre un 35 % y un 47 % de las insolvencias, y entre un 28 % y un 45 % de sus alertas son correctas. Si en el paper se priorizan los costes de un falso negativo, el umbral debería elegirse según esos costes.

### 11.5 Frentes de Pareto (rendimiento frente a velocidad)

Modelos no dominados, ordenados de más rápido a mejor rendimiento:

- **h1**: AUC CV vs predicción: GLM_1 → XGBoost_2 → XGBoost_3 → GBM_3 → GBM_2 → GBM_1 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1; AUCPR test vs predicción: GLM_1 → XGBoost_2 → XGBoost_3 → GBM_3 → GBM_1 → StackedEnsemble_AllModels_1; AUCPR test vs entrenamiento: GLM_1 → XGBoost_3 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1
- **h3**: AUC CV vs predicción: XGBoost_3 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1; AUCPR test vs predicción: XGBoost_3 → GBM_1 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1; AUCPR test vs entrenamiento: StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1
- **h5**: AUC CV vs predicción: GLM_1 → XGBoost_3 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1; AUCPR test vs predicción: GLM_1 → XGBoost_3 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1; AUCPR test vs entrenamiento: GLM_1 → StackedEnsemble_BestOfFamily_1 → StackedEnsemble_AllModels_1

Lectura:
- **XGBoost_3** es el mejor compromiso en h3 y h5: AUCPR en test de 0,29 y 0,41, frente a 0,34 y 0,45 del ensemble, y predice unas **9–14 veces más rápido** (0,006–0,008 frente a 0,065–0,109 ms/fila).
- En h1, los GBM y XGBoost también forman parte del frente, pero el ensemble saca más ventaja (0,231 frente a 0,210 de GBM_1).
- GLM solo aparece en el frente porque es el más rápido. Su rendimiento es muy inferior.

### 11.6 Importancia de variables (mejor modelo de árboles)

- **h1 (GBM_2)**: F21_1 (27.7%), F28_1 (15.3%), F20_1 (7.4%), F22_1 (3.0%), F37_2 (2.5%), F36_2 (1.5%), F25_1 (1.5%), F42_1 (1.0%) | por variable base: F21 29.2%, F28 18.4%, F20 8.6%, F37 4.1%, F22 3.4%, N12 2.9%
- **h3 (XGBoost_3)**: F27_1 (24.0%), F25_1 (5.0%), F41_1 (4.3%), F29_1 (3.6%), F28_1 (3.0%), F37_1 (2.5%), F20_1 (2.3%), N15.No (2.1%) | por variable base: F27 27.5%, F25 6.1%, F41 6.0%, N2 5.4%, F29 4.7%, F28 3.9%
- **h5 (XGBoost_3)**: F27_1 (14.5%), F25_1 (10.3%), F29_1 (5.1%), F28_1 (3.3%), F37_1 (2.9%), N2_1 (2.3%), F41_1 (2.3%), F36_1 (2.2%) | por variable base: F27 18.5%, F25 11.8%, N2 7.0%, F29 6.9%, F28 4.4%, F41 4.3%

Ahora **dominan ratios financieros del último año**, coherentes con la teoría:
- h1: **rentabilidad de explotación (F21)**, gastos financieros / activo total (F28) y rentabilidad económica (F20).
- h3 y h5: **carga financiera = (capital + intereses) / ventas (F27)**, cobertura de intereses EBIT / gastos financieros (F25), gastos financieros / ventas (F29) y liquidez (F36 y F37).
- Las variables no financieras que siguen pesando son el número de empleados (N2), el tamaño (N8), los órganos sociales (N12) y **empresa familiar (N15)**.

⚠️ **N15 merece revisarse.** "Familiar = Sí" aparece en el 30,7 % de las filas de solventes, el 15,2 % de las insolventes con fecha y solo el 3,5 % de las insolventes sin fecha. Que los dos grupos de insolventes difieran tanto entre sí apunta a un posible artefacto de la fuente, como en N11. Aun así, también hay un efecto real conocido en la literatura (las empresas familiares sobreviven más). No se ha eliminado.

### 11.7 Gráficos generados

- **h1** (mejor modelo de árboles: GBM_2): [GBM_2_learning_curve.png](runs/leakfree_v3_h1/plots/GBM_2_learning_curve.png), [GBM_2_pdp_F20_1.png](runs/leakfree_v3_h1/plots/GBM_2_pdp_F20_1.png), [GBM_2_pdp_F21_1.png](runs/leakfree_v3_h1/plots/GBM_2_pdp_F21_1.png), [GBM_2_pdp_F22_1.png](runs/leakfree_v3_h1/plots/GBM_2_pdp_F22_1.png), [GBM_2_pdp_F28_1.png](runs/leakfree_v3_h1/plots/GBM_2_pdp_F28_1.png), [GBM_2_pdp_F37_2.png](runs/leakfree_v3_h1/plots/GBM_2_pdp_F37_2.png), [GBM_2_shap_row_insolvent.png](runs/leakfree_v3_h1/plots/GBM_2_shap_row_insolvent.png), [GBM_2_shap_summary.png](runs/leakfree_v3_h1/plots/GBM_2_shap_summary.png), [GBM_2_varimp.png](runs/leakfree_v3_h1/plots/GBM_2_varimp.png), [leader_learning_curve.png](runs/leakfree_v3_h1/plots/leader_learning_curve.png), [metalearner_std_coef.png](runs/leakfree_v3_h1/plots/metalearner_std_coef.png), [model_correlation.png](runs/leakfree_v3_h1/plots/model_correlation.png), [pareto_auc_cv_vs_predict_time.png](runs/leakfree_v3_h1/plots/pareto_auc_cv_vs_predict_time.png), [pareto_aucpr_test_vs_predict_time.png](runs/leakfree_v3_h1/plots/pareto_aucpr_test_vs_predict_time.png), [pareto_aucpr_test_vs_training_time.png](runs/leakfree_v3_h1/plots/pareto_aucpr_test_vs_training_time.png), [varimp_heatmap.png](runs/leakfree_v3_h1/plots/varimp_heatmap.png)
- **h3** (XGBoost_3): [XGBoost_3_learning_curve.png](runs/leakfree_v3_h3/plots/XGBoost_3_learning_curve.png), [XGBoost_3_pdp_F25_1.png](runs/leakfree_v3_h3/plots/XGBoost_3_pdp_F25_1.png), [XGBoost_3_pdp_F27_1.png](runs/leakfree_v3_h3/plots/XGBoost_3_pdp_F27_1.png), [XGBoost_3_pdp_F28_1.png](runs/leakfree_v3_h3/plots/XGBoost_3_pdp_F28_1.png), [XGBoost_3_pdp_F29_1.png](runs/leakfree_v3_h3/plots/XGBoost_3_pdp_F29_1.png), [XGBoost_3_pdp_F41_1.png](runs/leakfree_v3_h3/plots/XGBoost_3_pdp_F41_1.png), [XGBoost_3_shap_row_insolvent.png](runs/leakfree_v3_h3/plots/XGBoost_3_shap_row_insolvent.png), [XGBoost_3_shap_summary.png](runs/leakfree_v3_h3/plots/XGBoost_3_shap_summary.png), [XGBoost_3_varimp.png](runs/leakfree_v3_h3/plots/XGBoost_3_varimp.png), [leader_learning_curve.png](runs/leakfree_v3_h3/plots/leader_learning_curve.png), [metalearner_std_coef.png](runs/leakfree_v3_h3/plots/metalearner_std_coef.png), [model_correlation.png](runs/leakfree_v3_h3/plots/model_correlation.png), [pareto_auc_cv_vs_predict_time.png](runs/leakfree_v3_h3/plots/pareto_auc_cv_vs_predict_time.png), [pareto_aucpr_test_vs_predict_time.png](runs/leakfree_v3_h3/plots/pareto_aucpr_test_vs_predict_time.png), [pareto_aucpr_test_vs_training_time.png](runs/leakfree_v3_h3/plots/pareto_aucpr_test_vs_training_time.png), [varimp_heatmap.png](runs/leakfree_v3_h3/plots/varimp_heatmap.png)
- **h5** (XGBoost_3): [XGBoost_3_learning_curve.png](runs/leakfree_v3_h5/plots/XGBoost_3_learning_curve.png), [XGBoost_3_pdp_F25_1.png](runs/leakfree_v3_h5/plots/XGBoost_3_pdp_F25_1.png), [XGBoost_3_pdp_F27_1.png](runs/leakfree_v3_h5/plots/XGBoost_3_pdp_F27_1.png), [XGBoost_3_pdp_F28_1.png](runs/leakfree_v3_h5/plots/XGBoost_3_pdp_F28_1.png), [XGBoost_3_pdp_F29_1.png](runs/leakfree_v3_h5/plots/XGBoost_3_pdp_F29_1.png), [XGBoost_3_pdp_F37_1.png](runs/leakfree_v3_h5/plots/XGBoost_3_pdp_F37_1.png), [XGBoost_3_shap_row_insolvent.png](runs/leakfree_v3_h5/plots/XGBoost_3_shap_row_insolvent.png), [XGBoost_3_shap_summary.png](runs/leakfree_v3_h5/plots/XGBoost_3_shap_summary.png), [XGBoost_3_varimp.png](runs/leakfree_v3_h5/plots/XGBoost_3_varimp.png), [leader_learning_curve.png](runs/leakfree_v3_h5/plots/leader_learning_curve.png), [metalearner_std_coef.png](runs/leakfree_v3_h5/plots/metalearner_std_coef.png), [model_correlation.png](runs/leakfree_v3_h5/plots/model_correlation.png), [pareto_auc_cv_vs_predict_time.png](runs/leakfree_v3_h5/plots/pareto_auc_cv_vs_predict_time.png), [pareto_aucpr_test_vs_predict_time.png](runs/leakfree_v3_h5/plots/pareto_aucpr_test_vs_predict_time.png), [pareto_aucpr_test_vs_training_time.png](runs/leakfree_v3_h5/plots/pareto_aucpr_test_vs_training_time.png), [varimp_heatmap.png](runs/leakfree_v3_h5/plots/varimp_heatmap.png)
