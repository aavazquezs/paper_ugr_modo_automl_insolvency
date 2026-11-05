# Descripción de los elementos modificados

## Resumen
Se ha corregido el riesgo de *data leakage* en el pipeline de predicción de insolvencia de pymes, pasando de un *split* aleatorio por filas sobre datos ya balanceados a un enfoque *leak-free* con *group split* por empresa.

## Cambios realizados

### 1. Scripts nuevos (en `scripts/`)
| Archivo | Descripción |
| --- | --- |
| `02_fix_split_leakfree.py` | Realiza *group split* por `CIF` usando `train_test_split` sobre empresas únicas (semilla 1234, test_size=0.3). Genera listas de CIFs train/test y crea splits iniciales. |
| `03_undersample_train_only.py` / `balance_fix2.py` (ejecutados) | Realiza undersampling aleatorio únicamente sobre el conjunto de entrenamiento para balancear clases. El conjunto de test se mantiene sin balancear. |
| `make_test_h3.py`, `make_test_unbal_fast.py` | Construyen ventanas *firm-history* para los CIFs asignados a test a partir de `raw_data.csv`, generando test con prevalencia natural. |

### 2. Datos generados (en `data/`)
| Ruta | Descripción |
| --- | --- |
| `data/splits/train_cifs_h{1,3,5}.json` | Lista de `CIF` únicos asignados al conjunto de entrenamiento (sin solapamiento con test). |
| `data/splits/test_cifs_h{1,3,5}.json` | Lista de `CIF` únicos asignados al conjunto de test. |
| `data/final/train_balanced_leakfree_h{1,3,5}.csv` / `.parquet` | Conjunto de entrenamiento balanceado (undersampling solo en train), con *group split* por `CIF`. |
| `data/final/test_unbalanced_leakfree_h{1,3,5}.csv` / `.parquet` | Conjunto de test sin balancear, con prevalencia natural. Construido desde datos crudos para los CIFs de test. |

### 3. Notebooks modificados
| Archivo | Modificaciones |
| --- | --- |
| `H10h1.ipynb`, `H10h3.ipynb`, `H10h5.ipynb` | Eliminado `split_frame` sobre datos balanceados. Ahora cargan `train_balanced_leakfree_h{h}.csv` y `test_unbalanced_leakfree_h{h}.csv`. Se eliminan identificadores (`CIF`, `window_id`, `ref_year`, `horizon`) de predictores. Se usa `sort_metric="auc_pr"` y semilla 1234. La evaluación se realiza sobre test con prevalencia natural. |

### 4. Documentación
| Archivo | Descripción |
| --- | --- |
| `README_LEAKAGE_FIX.md` | Documentación resumida de la metodología leak-free aplicada. |
| `DESCRIPCION_MODIFICACIONES.md` | Este documento con descripción detallada de los elementos modificados. |

## Verificación
- **Sin solapamiento de CIF**: Verificado para h1,3,5 (intersección vacía entre train/test).
- **Test con prevalencia natural**: h1: {0: 0.914, 1: 0.086}, h3: {0: 0.930, 1: 0.070}, h5: {0: 0.914, 1: 0.086}.
- **Balanceo solo en train**: Undersampling aplicado únicamente al conjunto de entrenamiento.
- **Semillas fijas**: 1234 en split, undersampling y configuración de H2O AutoML.
