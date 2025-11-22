# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import sys, pandas as pd
sys.path.insert(0, "/workspace/src")
from feat_picker import pick_features
df = pd.read_csv("/workspace/data/data_processed/Complete_DataSet_raw_clim.csv")
target_col = "CP"
feats = pick_features(df, target_col)
X = df[feats].values
y = df[target_col].values
print(f"| feats={len(feats)} | X={X.shape} | y={y.shape}")
