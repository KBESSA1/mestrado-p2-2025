# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de Gradient Boosting com seleção de features (FS) via XGBoost. Foca em cenários finais com features reduzidas mais estáveis.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def main():
    data_path = Path("/workspace/data/data_raw/Complete_DataSet.csv")
    df = pd.read_csv(data_path)

    target_col = "TDN_based_ADF"
    date_col = "Date"

    print("[INFO] Lendo CSV de", data_path)
    before = len(df)
    df = df.dropna(subset=[date_col, target_col])
    print(f"[INFO] Linhas após remover nulos em datas e alvo ({target_col}): {len(df)} (removidas {before-len(df)})")

    df[date_col] = pd.to_datetime(df[date_col])

    # ----------------------
    # Delta_days
    # ----------------------
    print("[INFO] Verificando/Calculando Delta_days...")
    if "Delta_days" in df.columns:
        print("[INFO] Usando coluna Delta_days já existente")
    else:
        image_cols = [c for c in df.columns if "image" in c.lower() or "sensing" in c.lower()]
        if image_cols:
            img_col = image_cols[0]
            df[img_col] = pd.to_datetime(df[img_col])
            df["Delta_days"] = (df[img_col] - df[date_col]).abs().dt.days
            print(f"[INFO] Delta_days calculado a partir de {img_col}")
        else:
            raise RuntimeError("Não há coluna Delta_days nem coluna de data de imagem para calcular Delta_days")

    # ----------------------
    # Filtro D5
    # ----------------------
    df_d5 = df[df["Delta_days"] <= 5].copy()
    print(f"[INFO] Nº de amostras após filtro D5: {len(df_d5)}")

    # ----------------------
    # Features estáveis via XGB (TDN) - vamos ler o CSV de estabilidade
    # ----------------------
    stab_path = Path("/workspace/reports/feature_importance/feature_importance_xgb_stability_TDN_based_ADF.csv")
    print(f"[INFO] Lendo estabilidade de features de {stab_path}")
    stab = pd.read_csv(stab_path)

    stable = stab[stab["freq_in_topK"] >= 0.75]["feature"].tolist()
    if not stable:
        stable = (
            stab.sort_values("mean_importance", ascending=False)["feature"]
                .head(15)
                .tolist()
        )
        print("[WARN] Nenhuma feature com freq_in_topK>=0.75; usando top-15 por importância média.")
    print("[INFO] Features estáveis (antes de remover clima/leak):", stable)

    # Colunas de clima para remover em noclim
    climate_cols = [
        "TEMP_MAX", "TEMP_MIN", "RAIN",
        "RAD_SOL", "Radiative_Dif_AVG", "Radiative_Direct_AVG",
        "Longwave_Rad_AVG", "TP_SFC_AVG", "PRES_ATM",
        "EVAPOT", "WIND_SPD", "Wind_Dir",
        "HUM_REL", "PPFD",
    ]

    # Colunas de leak em relação a TDN_based_ADF
    leak_cols = ["ADF", "TDN_based_NDF", "CP"]

    drop_cols = set(climate_cols + leak_cols)

    features = [f for f in stable if f not in drop_cols]
    print("[INFO] Ignorando colunas de clima e leak (noclim):", [c for c in stable if c in drop_cols])
    print(f"[INFO] Nº de features numéricas candidatas em X: {len(features)}")
    print("[INFO] Algumas features:", features[:10])

    X = df_d5[features].astype(float)
    y = df_d5[target_col].astype(float)
    print(f"[INFO] Shape de X: {X.shape}, y: {y.shape}")

    # ----------------------
    # LODO por data
    # ----------------------
    dates = sorted(df_d5[date_col].unique())
    print(f"[INFO] Nº de datas (folds LODO): {len(dates)}")

    metrics_rows = []
    oof_list = []

    for fold_idx, d in enumerate(dates, start=1):
        test_mask = df_d5[date_col] == d
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        print(f"[FOLD {fold_idx:02d}] Date={d.date()} | n_train={len(y_train)} | n_test={len(y_test)}")

        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        print(f"[FOLD {fold_idx:02d}] Date={d.date()} | R2={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")

        metrics_rows.append({
            "fold": fold_idx,
            "date": d,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        })

        oof_list.append(pd.DataFrame({
            "Date": df_d5.loc[test_mask, date_col].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }))

    metrics_df = pd.DataFrame(metrics_rows)
    oof_df = pd.concat(oof_list, axis=0)

    # ----------------------
    # Global OOF
    # ----------------------
    all_true = oof_df["y_true"].values
    all_pred = oof_df["y_pred"].values
    r2_global = r2_score(all_true, all_pred)
    mse_global = mean_squared_error(all_true, all_pred)
    rmse_global = mse_global ** 0.5
    mae_global = mean_absolute_error(all_true, all_pred)

    print(f"\n[GLOBAL OOF - GB_D5_TDN_based_ADF_NOCLIM_FS] R2={r2_global:.3f} | RMSE={rmse_global:.3f} | MAE={mae_global:.3f}")

    metrics_df.loc[len(metrics_df)] = {
        "fold": "GLOBAL_OOF",
        "date": pd.NaT,
        "n_train": len(all_true),
        "n_test": 0,
        "R2": r2_global,
        "RMSE": rmse_global,
        "MAE": mae_global,
    }

    out_dir = Path("/workspace/reports/baseline_experimento_UFMS")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "baseline_experimento_UFMS_D5_TDN_based_ADF_noclim_gb_fs_metrics.csv"
    oof_path = out_dir / "baseline_experimento_UFMS_D5_TDN_based_ADF_noclim_gb_fs_oof.csv"

    metrics_df.to_csv(metrics_path, index=False)
    oof_df.to_csv(oof_path, index=False)

    print(f"[OK] Métricas por fold + GLOBAL (GB D5_TDN_based_ADF_noclim_gb_fs) salvas em: {metrics_path}")
    print(f"[OK] Predições OOF (GB D5_TDN_based_ADF_noclim_gb_fs) salvas em: {oof_path}")


if __name__ == "__main__":
    main()
