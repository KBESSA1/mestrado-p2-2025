# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de experimento LODO para TDN_based_ADF (digestibilidade estrutural), usando XNet (Cauchy network), janela Δ7 dias (D7), com clima acoplado ao S2. Faz parte do painel de baselines oficiais do mestrado (UFMS).
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===================== CONFIG =====================

CSV_PATH = "/workspace/data/data_raw/Complete_DataSet.csv"
OUT_DIR = Path("/workspace/reports/baseline_experimento_UFMS")

TARGET_COL = "TDN_based_ADF"
CLIMATE_COLS = ["TEMP_MAX", "TEMP_MIN", "RAIN"]
DELTA_MAX_DAYS = 7  # D7
LEAK_COLS = ["ADF", "TDN_based_NDF", "CP"]   # não podem entrar

SEED = 42

# ===================== XNET (Cauchy) =====================

class CauchyActivation(nn.Module):
    """
    phi(x) = lambda1 * x / (x^2 + d^2) + lambda2 / (x^2 + d^2)
    com d > 0 via softplus
    """
    def __init__(self, dim):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.zeros(dim))
        self.lambda2 = nn.Parameter(torch.zeros(dim))
        self.d_raw = nn.Parameter(torch.zeros(dim))
        self.softplus = nn.Softplus()
        self.eps = 1e-6

    def forward(self, x):
        d = self.softplus(self.d_raw) + self.eps
        denom = x * x + d * d
        return self.lambda1 * x / denom + self.lambda2 / denom


class XNetRegressor(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.fc_in = nn.Linear(d_in, hidden)
        self.act = CauchyActivation(hidden)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.fc_in(x)
        z = self.act(z)
        y = self.fc_out(z)
        return y


# ===================== PREPARO DOS DADOS =====================

def prepare_data():
    print(f"[INFO] Lendo CSV de {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Garantir colunas de data
    for col in ["Date", "Satellite_Images_Dates"]:
        if col not in df.columns:
            raise ValueError(f"Coluna de data '{col}' não encontrada no CSV.")
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remover nulos em datas e alvo
    before = len(df)
    df = df.dropna(subset=["Date", "Satellite_Images_Dates", TARGET_COL]).copy()
    after = len(df)
    print(
        f"[INFO] Linhas após remover nulos em datas e alvo ({TARGET_COL}): "
        f"{after} (removidas {before - after})"
    )

    # Delta em dias
    print("[INFO] Calculando Delta_days...")
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).abs().dt.days

    # Filtro D7
    df = df[df["Delta_days"] <= DELTA_MAX_DAYS].copy()
    print(f"[INFO] Nº de amostras após filtro D7: {len(df)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remover alvo, Delta_days e leaks; manter clima
    base_drop = [TARGET_COL, "Delta_days"] + LEAK_COLS
    feature_cols = [c for c in numeric_cols if c not in base_drop]

    print(f"[INFO] Colunas de clima usadas (D7 CLIM): {CLIMATE_COLS}")
    print(f"[INFO] Removendo colunas 'leak' do X: {LEAK_COLS}")
    print(f"[INFO] Nº de features numéricas (com clima, sem leak): {len(feature_cols)}")
    print(f"[INFO] Algumas features: {feature_cols[:10]}")

    return df, feature_cols


# ===================== LODO XNET =====================

def run_lodo_xnet():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device usado pelo XNet: {device}")

    df, feature_cols = prepare_data()

    dates_unique = sorted(df["Date"].unique())
    print(f"[INFO] Nº de datas (folds LODO): {len(dates_unique)}")

    all_metrics = []
    oof_records = []

    for i, d in enumerate(dates_unique, start=1):
        print(f"[FOLD {i:02d}] Date={d.date()}")

        test_mask = df["Date"] == d
        train_mask = ~test_mask

        df_train = df.loc[train_mask].copy()
        df_test = df.loc[test_mask].copy()

        X_train_raw = df_train[feature_cols].values.astype(np.float32)
        y_train = df_train[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

        X_test_raw = df_test[feature_cols].values.astype(np.float32)
        y_test = df_test[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        X_train_t = torch.from_numpy(X_train).to(device)
        y_train_t = torch.from_numpy(y_train).to(device)
        X_test_t = torch.from_numpy(X_test).to(device)

        ds_train = TensorDataset(X_train_t, y_train_t)
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)

        model = XNetRegressor(d_in=X_train.shape[1], hidden=64).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        max_epochs = 500
        model.train()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for xb, yb in dl_train:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(ds_train)

        model.eval()
        with torch.no_grad():
            y_pred_t = model(X_test_t).cpu().numpy().reshape(-1)

        y_true = y_test.reshape(-1)
        y_pred = y_pred_t

        r2 = r2_score(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"[FOLD {i:02d}] Date={d.date()} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        all_metrics.append(
            {
                "fold": i,
                "date": d,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

        for dt, yt, yp in zip(df_test["Date"], y_true, y_pred):
            oof_records.append(
                {
                    "Date": dt,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "fold": i,
                }
            )

    oof_df = pd.DataFrame(oof_records)
    y_true_all = oof_df["y_true"].values
    y_pred_all = oof_df["y_pred"].values

    r2_global = r2_score(y_true_all, y_pred_all)
    rmse_global = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
    mae_global = mean_absolute_error(y_true_all, y_pred_all)

    print(
        f"\n[GLOBAL OOF - XNET_D7_TDN_CLIM_NOADF] "
        f"R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.loc[len(metrics_df)] = {
        "fold": "GLOBAL",
        "date": pd.NaT,
        "n_train": np.nan,
        "n_test": int(oof_df.shape[0]),
        "R2": r2_global,
        "RMSE": rmse_global,
        "MAE": mae_global,
    }

    metrics_path = OUT_DIR / "baseline_experimento_UFMS_D7_TDN_clim_xnet_metrics.csv"
    oof_path = OUT_DIR / "baseline_experimento_UFMS_D7_TDN_clim_xnet_oof.csv"

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(
        f"\n[OK] Métricas por fold + GLOBAL (XNet D7 TDN CLIM, sem ADF/TDN_NDF/CP) salvas em: {metrics_path}"
    )
    print(
        f"[OK] Predições OOF (XNet D7 TDN CLIM, sem ADF/TDN_NDF/CP) salvas em: {oof_path}"
    )


if __name__ == "__main__":
    run_lodo_xnet()
