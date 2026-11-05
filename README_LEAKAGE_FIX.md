# Corrección de data leakage

Se ha corregido el pipeline para evitar fugas de información:

## Metodología aplicada
1. **Split a nivel empresa (group split)**: División por `CIF` con `train_test_split` sobre empresas únicas (`random_state=1234`, test_size=0.3). Garantizado sin solapamiento entre train/test.
2. **Test con prevalencia natural**: Las ventanas de test se construyen desde `raw_data.csv` para los `CIF` asignados a test, manteniendo la distribución original de `Y` (no balanceado).
3. **Balanceo solo en train**: Undersampling aleatorio únicamente sobre el conjunto de entrenamiento, con semilla fija. El test permanece sin balancear.
4. **Sin fuga en preprocesamiento**: Cualquier transformación debe ajustarse solo con train y aplicarse a test (aplicado en el nuevo flujo).
5. **Semillas fijas**: `1234` en split, undersampling y H2O AutoML.

## Archivos generados
- `data/splits/train_cifs_h{1,3,5}.json`, `data/splits/test_cifs_h{1,3,5}.json`
- `data/final/train_balanced_leakfree_h{1,3,5}.csv/parquet`
- `data/final/test_unbalanced_leakfree_h{1,3,5}.csv/parquet`

## Notebooks actualizados
Los notebooks `H10h1.ipynb`, `H10h3.ipynb`, `H10h5.ipynb` ahora cargan los splits leak-free sin usar `split_frame` sobre datos balanceados globalmente.
