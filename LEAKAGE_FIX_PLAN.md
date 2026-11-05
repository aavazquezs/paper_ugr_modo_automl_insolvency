#### Plan de acción para corregir el riesgo de *data leakage* en la predicción de insolvencia de pymes españolas

**Objetivo:** Reestructurar el proceso de generación de datasets y modelado para garantizar que no exista fuga de información (*data leakage*) a lo## Commit 1: plan de corrección de data leakage

# Plan de acción para corregir el riesgo de *data leakage* en la
# predición de insolvencia de pymes españolas

**Objetivo:** Reestructurar
nivel empresa-histórico (*firm-history windows*), asegurando
un *split* correcto, un preprocesamiento sin fuga, un balanceo
únicamente sobre *train*, y una evaluación realista sobre *test* con
prevalencia natural.

---

## 1. Problemas identific# PR Title: Fix data leakage in SME insolvency prediction pipeline

**Objetivo:** Corregir el proceso de generación de conjuntos de datos y
modelado para evitar fuga de información (*data leakage*),
garantizando un split a nivel empresa, preprocesamiento sin fuga,
undersampling únicamente en entrenamiento y evaluación con prevalencia
natural.

## 1. Problemas identificados

| Problema | Riesgo | Explicación |
|
:-- | :-- | :-- |
| *Split* aleatorio por filas | Muy alto | Los notebooks `H10h*.ipynb` usan `split_frame(ratios=[0.7], seed=1234) sobre los CSVs ya balanceados. Al existir múltiples ventanas por `CIF`, puede haber solapamiento entre `train` y `test. | |
| Undersampling antes del split | Muy alto | Los archivos `datasets/10_to_predict_*_balanced.csv` ya están balanceados. Se ha seleccionado qué ventanas solventes mantienen mirando todo el dataset, incluyendo el `test. | |
| Imputación global/k-NN antes del split | Alto | Si la imputación global con k-NN se realizó sobre el dataset completo antes de dividir, el `test` influye en los valores imputados del `train. | |
| Autocorrelación entre ventanas | Alto | Las ventanas desplazadas de la misma empresa están fuertemente correlacionadas. Un *split* a nivel fila es incorrecto, incluso si no hubiera solapamiento por CIF. |

---

## 2. Procedimiento correcto a implementar

### Paso 1: Trabajar desde los datos crudos sin balancear
- **Fuente:** `datasets/raw_data.csv` (o equivalente original). Los CSVs balanceados actuales no deben usarse como punto de partida para el nuevo split.
- **Crear todas las *firm-history windows*** válidas para cada horizonte `h` en {1,3,5} con histórico de 10 años y estado futuro conocido (`Y` in {0,1}) en `t+h`.
- **Mantener el DataFrame sin balancear** para el proceso de split. Guardar esta versión como `windows_h{h}_unbalanced.csv`.

### Paso 2: Split a nivel empresa (*group split*) y temporal (*time-aware*)
- **Nivel empresa:** Dividir por `CIF`. Todas las ventanas de una misma CIF deben ir exclusivamente a `train` o a `test. Definir grupos = `CIF`.`
- **Temporal:** Siempre que sea posible, asignar ventanas con periodo histórico/predicción *anteriores* al periodo del test as a *out-of-time holdout*. Si no se puede garantizar completamente temporal por falta de cobertura, al menos asegurar el *group split* por CIF. Documentar esta decisión.
- **Ratio recomendado:** 70% empresas/ventanas en `train`, 30% in `test` (o 80/20) con semillas fijas. Especificar el número de *CIFs únicos* en cada split.
- **Validación interna:** Dentro de `train`, H2O AutoML podrá hacer su CV interno, pero debe hacerse respetando el agrupamiento por `CIF` (o simplemente sobre `train` debidamente procesado). 

### Paso 3: Preprocesamiento únicamente sobre *train*
Todas las transformaciones deben ajustarse exclusivamente on `train` y aplicarse posteriormente a `test:`

| Transformación | Implementación | Notas |
| :-- | :-- | :-- |
| Imputación a nivel empresa | Mediana (numéricas) / moda (categórídes) por `CIF` usando únicamente las ventanas/observaciones de `train. | No usar información de `test` ni del histórico futuro. |
| Eliminación de variables con alta tasa de perdidos | Basarse en el análisis de *missingness* hecho sobre `train` (variables `N11`, `F26`, `F32`, etc.). Aplicar al `test. | Eliminarlas después del análisis solo sobre train. |
| Imputación global/k-NN | Ajustar el imputador k-NN (k=5) únicamente con las observaciones de `train` as matrix. Aplicar la función *transform* a `test. | Es clave evitar usar vecinos del `test`. |
| Eliminación por redundancia | Determinar las variables redundantes sobre `train (ej. `N9` vs `N10`). Aplicar al `test. | |
| Otras transformaciones (escalado/encoding) | Ajustar cualquier estándarizador, encoder or catboost-like encoder sobre `train. Aplicar a test. | |

### Paso 4: Balanceo únicamente sobre *train*
- **Undersampling aleatorio** de las ventanas solventes (`Y=0`) únicamente sobre el conjunto *train*, hasta alcanzar una relación 1:1 con las ventanas insolventes (`Y=1`) presentes in `train. | |
- **Mantener `test` sin balancear** con su **prevalencia natural**. Esto es imprescindible para evaluar métricas realistas: AUC, AUC-PR, precisión, recall, F1, matrices de confusión at sensible threshold, calibración, etc. 
- **Repetibilidad:** Usar semillas fijas (*seed*) para el undersampling, y reportar el número de ventanas *train/unbalanced*, *train/balanced* and *test/unbalanced* así como el ratio de insolventes in `test`.

### Paso 5: Modelado y evaluación
- **H2O AutoML**: Entrenar únicamente sobre `train/balanced` (o sobre `train` si se prefiere, pero la elección de balancear solo en train es compatible. El balanceo es para facilitar la búsqueda, pero la evaluación debe hacerse en `test` unbalanced). Es importante no pasar `test` al proceso de búsqueda o selección. 
- **Selección de modelos:** Usar el esquema Pareto + interpretabilidad tal como está en el paper, pero justificar explicando la relación entrega *train* a *test*.
- **Evaluación final:** Reportar métricas sobre `test` con prevalencia natural. Considerar AUC-PR como principal, junto *confusion matrices* at thresholds óptimos por costo/utility or Business risk-related thresholds.
- **Baselines:** Evaluar *logistic regression*, *decision tree* (CART), *random forest* y el enfoque tipo Navarro-Galera (decision tree manually tuned) en **exactamente las mismas particiones** (*group+temporal*) con el mismo preprocesamiento regla a partir de `train`.

---

## 2.1. Referencias de código/archivos existentes
| Archivo/Directorio | Uso |
| :-- | :-- |
| `/media/datos/WORKSPACE/DOCTORADO/datasets/raw_data.csv` | Fuente original sin balancear. |
| `/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency/datasets/`| CSVs actualmente balanceados (solo für Modelado, no für Nuevo split) | |
| `/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency/H10h*.ipynb` | Notebooks de H2O AutoML (fase de modelado). Será necesario reestructurarlos. |

---

## 3. Acciones concretas a implementar para un agente

### Acción 1: Crear script de creación de ventanas
- **Archivo:** `scripts/create_windows.py`
- **Descripción:** Crear *firm-history windows* de 10 años + horizonte h, manteniendo `CIF`, `Year` (inicio/final de ventana), el estado `Y` futuro, y todas las variables `N*` y `F*`. Agregar una columna `window_id`.
- **Salida:** `data/windows_h{h}_unbalanced.parquet` / `.csv`.

### Acción 2: Crear script de split por grupo y temporal
- **Archivo:** `scripts/split_data.py`
- **Descripción:** Obtener lista de `CIFs únicos`, hacer *group split* por `CIF` asegurando solapamiento cero, intentar *time-aware split* (buscar año de referencia por *CIF* o por ventana). Crear índices, guardar `train_cifs.json`, `test_cifs.json`.
- **Salida:** `data/train_windows_h{h}.parquet`, `data/test_windows_h{h}.parquet` (sin balancear).

### Acción 3: Crear script de preprocesamiento sin fuga
- **Archivo:** `scripts/preprocess_leakfree.py`
- **Descripción:** Ajustar: *missingness* analysis over *train, imputación a nivel empresa (medianas/modas por `CIF` in `train), estimadores globales, eliminación variables, k-NN (fit *only* *train), redundancias. Aplicar pipeline *transform* to *test. | |
- **Salida: *train & test` sin balancear tras preprocesamiento, preservando *CIF* (para el balanceo/verification).` *{h}_proc_unbalanced.parquet` |

### Acción 4: Crear script de balanceo solo en *train*
- **Archivo:** `scripts/balance_train_only.py`
- **Descripción:** Hacer *undersampling* aleatorio *only* *train proc unbalanced` con `seed=1234. \tal Mantener *test proc unbalanced. Save \`train_balanced_h{h}.parquet`, \`test_unbalanced_h{h}.parquet`.
- **Salida:** Balanceados/No balanceados, prevalencia naturl in *test.

### Acción 5: Reestructurar notebooks de H2O AutoML
- **Archivos: ** `H10h1.ipynb`, `H10h3.ipynb`, `H10h5.ipynb` 
- **Descripción:** Cambiar importación: cargar `train_balanced_h{h}.parquet` . *group split? H2O AutoML internal CV sobre *train* (pero debe ser *group-aware. H2O no always group-aware by default, so manual strategy or use folds by *CIF*). Use *test_unbalanced_h{h}.parquet* für final evaluation. Eliminar CIF desde ponerhar como predictor (o mantener para contrôle). Report AUC-PR, confusióng matrix, calibración in *test* unbalanced. Add hyperparam settings (max_models, max_runtime_secs, nfolds, seed, sort_metric) explicitly. |
- **Requisitos:** *seed* fijas, *folds* por / senza solapamiento de CIF, *leaderboard_frame* y *validation_frame* correctamente usados.

---

## 4. Elementos a documentar en el paper (respuestas a revisores)
| Punto del revisor | Acción en el manuscript | 
| :-- | :-- | 
| Unidad de análisis | Especificar que la unidad es * *firm-history window* (una fila = ventana de 10 años de una empresa). Indicar número de *CIFs únicos* in *train* y *test*. | |
| Split y *leakage* | Describir el *group split* por *CIF* + *time-aware*, que el preprocesamiento se ajustó únicamente con `train`, y el undersampling se hizo solo en `train`. El `test` se mantuvo con prevalencia natural. | |
| Prevalencia natural | Reportar métricas sobre *test* unbalanced (AUC-PR, precisión, recall, F1, matriz de confusión, calibración). | |
| AutoML settings | Reportar `seed`, `max_models`, `max_runtime_secs`, `nfolds`, algoritmo/inclusión/exclusión, métricas usadas, fold construction. | |
| Undersampling seeds | Reportar estabilidad con múltiples semillas de undersampling *only* on *train* (media ± DE) sobre *test* unbalanced. | |
| Baselines | Evaluar baselines on **exactamente las mismas partículas\*\* (*CIF* split) con el pet preprocesamiento aplicando reglas de *train*. | 

---

## 5. Checklist de verificación para un agente
- [ ] **Grupo split:** Verificar que ningún *CIF* aparece en *train* y *test*. (Código: `set(train_cifs) & set(test_cifs) == empty` ) | |
- [ [ ] **Preprocesamiento sin fuga:** Verificar que los parámetros (medianas por *CIF*, media/moda globali adjusted with *train*, imputador k-NN ajustado) se ajustaron exclusivamente with *train. | |
- [ ] **Balanceo solo en train:** El *test* no está balanceado. Verificar ratio real en *test. | |
- [ ] **Unidad correct: *firm-history window*:** Documentar que una empresa puede aparecer múltiples veces, pero *NO* entre *train* and *test. | |
- [ ] **Información temporal:** Si es posible, describir el rango temporal de *train* / *test* as *out-of-time. | |
- [ ] **Semillas fijas:** *seed=1234` everywhere (split, undersampling, H2O AutoML) | |
- [ ] **Código abierto:** Los scripts generados, README.md opisante, và e publicaran a la *GitHub* URL (verificar *spaces* en URL). | |

---

vlpmpq#レ#le##eadk人
---

## Nota final
Avec *leakage* claro, el texto del paper afirma que se hizo a nivel empresa y temporal, pero esto **NO coincidente** avec le code. **It is imperative to reestructurarem seu código e experimento** para que coincide realmente. The agente deve implementar este plan completamente, não limitar-se apenas à correção do texto. 
