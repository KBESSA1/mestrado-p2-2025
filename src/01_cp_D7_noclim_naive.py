# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para CP (proteína bruta), usando baseline ingênuo (persistência), janela Δ7 dias (D7), apenas bandas/índices do S2, sem clima. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===========================
# CONFIGURAÇÃO DO EXPERIMENTO
# ===========================

CSV_PATH = Path("/workspace/data/data_raw/Complete_DataSet.csv")
OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")

DATE_COL = "Date"
SAT_COL = "Satellite_Images_Dates"
TARGET_COL = "CP"
MAX_DELTA_DAYS = 7

SUFFIX = "D7_CP_noclim_naive"  # identificador


# ============
# CARREGAR DADOS
# ============

print(f"[INFO] Lendo CSV de {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print("[INFO] Convertendo colunas de data...")
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df[SAT_COL] = pd.to_datetime(df[SAT_COL])

# Remover linhas sem alvo ou datas inválidas
df = df.dropna(subset=[TARGET_COL, DATE_COL, SAT_COL]).copy()

print("[INFO] Calculando Delta_days...")
df["Delta_days"] = (df[DATE_COL] - df[SAT_COL]).abs().dt.days

print(f"[INFO] Filtrando linhas com Delta_days <= {MAX_DELTA_DAYS}...")
df = df[df["Delta_days"] <= MAX_DELTA_DAYS].copy()

if df.empty:
    raise RuntimeError("Nenhuma linha restante após filtro de Delta_days. Verifique os dados!")

# Remover colunas de clima (noclim)
clim_cols = [c for c in df.columns if any(k in c.upper() for k in ["ERA5", "CHIRPS", "CLIM"])]
if clim_cols:
    print(f"[INFO] Removendo colunas de clima (noclim): {clim_cols}")
    df = df.drop(columns=clim_cols)

y = df[TARGET_COL].to_numpy(dtype=float)
groups = df[DATE_COL].values

print(f"[INFO] Nº de amostras após filtro D7: {len(y)}")

# ============
# LODO + NAIVE
# ============

logo = LeaveOneGroupOut()

oof_pred = np.zeros_like(y, dtype=float)
fold_metrics = []

print("[INFO] Iniciando LODO (baseline Naive)...")
for fold, (train_idx, test_idx) in enumerate(logo.split(y, y, groups), start=1):
    y_train, y_test = y[train_idx], y[test_idx]

    # Modelo Naive: sempre prever a média de y_train
    y_hat = np.full_like(y_test, fill_value=y_train.mean())

    oof_pred[test_idx] = y_hat

    r2 = r2_score(y_test, y_hat)
    mse = mean_squared_error(y_test, y_hat)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_hat)

    date_str = pd.to_datetime(groups[test_idx][0]).strftime("%Y-%m-%d")
    print(f"[FOLD {fold:02d}] Date={date_str} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

    fold_metrics.append(
        {
            "fold": fold,
            "date": date_str,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "n_test": len(test_idx),
        }
    )

# =====================
# MÉTRICAS GLOBAIS (OOF)
# =====================

r2_global = r2_score(y, oof_pred)
mse_global = mean_squared_error(y, oof_pred)
rmse_global = mse_global ** 0.5
mae_global = mean_absolute_error(y, oof_pred)

print("\n[GLOBAL OOF - NAIVE] R2={:.3f} | RMSE={:.3f} | MAE={:.3f}".format(
    r2_global, rmse_global, mae_global
))

fold_metrics.append(
    {
        "fold": "GLOBAL_OOF",
        "date": "ALL",
        "r2": r2_global,
        "rmse": rmse_global,
        "mae": mae_global,
        "n_test": len(y),
    }
)

metrics_df = pd.DataFrame(fold_metrics)

oof_df = pd.DataFrame(
    {
        "Date": df[DATE_COL].dt.strftime("%Y-%m-%d"),
        "target": y,
        "pred": oof_pred,
    }
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

metrics_path = OUT_DIR / f"baseline_experimento_UFMS_{SUFFIX}_metrics.csv"
oof_path = OUT_DIR / f"baseline_experimento_UFMS_{SUFFIX}_oof.csv"

metrics_df.to_csv(metrics_path, index=False)
oof_df.to_csv(oof_path, index=False)

print(f"\n[OK] Métricas por fold + GLOBAL (Naive) salvas em: {metrics_path}")
print(f"[OK] Predições OOF (Naive) salvas em: {oof_path}")
