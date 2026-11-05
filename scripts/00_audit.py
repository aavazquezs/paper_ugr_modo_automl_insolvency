#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path

ROOT = Path('/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency')
RAW = Path('/media/datos/WORKSPACE/DOCTORADO/datasets/raw_data.csv')

df = pd.read_csv(RAW)
print('raw shape', df.shape)
print(df.columns[:15])
print(df['CIF'].nunique(), 'unique CIF')
print(df['Year'].min(), df['Year'].max())
