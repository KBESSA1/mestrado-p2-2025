# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

CSV_PATH = "/workspace/data/data_raw/Complete_DataSet.csv"
OUT_DIR = Path("/workspace/reports/feature_importance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIMATE_COLS = ["TEMP_MAX", "TEMP_MIN", "RAIN"]

def prepare_dataset(target_col: str, delta_max_days: int, use_climate: bool):
    print(f"\n[INFO] === Cenário: target={target_col} | D{delta_max_days} | clim={use_climate} ===")
    df = pd.read_csv(CSV_PATH)

    # Datas
    for col in ["Date", "Satellite_Images_Dates"]:
        if col not in df.columns:
            raise ValueError(f"Coluna de data '{col}' não encontrada no CSV.")
        df[col] = pd.to_datetime(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Date", "Satellite_Images_Dates", target_col]).copy()
    after = len(df)
    print(f"[INFO] Linhas após remover nulos em datas e alvo ({target_col}): {after} (removidas {before - after})")

    # Delta_days e filtro D5/D7
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).abs().dt.days
    print("[INFO] Calculando Delta_days...")
    df = df[df["Delta_days"] <= delta_max_days].copy()
    print(f"[INFO] Nº de amostras após filtro D{delta_max_days}: {len(df)}")

    # Leak cols para TDN
    leak_cols = []
    if target_col == "TDN_based_ADF":
        leak_cols = ["ADF", "TDN_based_NDF", "CP"]
        leak_cols = [c for c in leak_cols if c in df.columns]
        if leak_cols:
            print(f"[INFO] Removendo colunas 'leak' do X: {leak_cols}")

    # Definir colunas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    drop_cols = [target_col, "Delta_days"]
    if not use_climate:
        drop_cols += CLIMATE_COLS
        print(f"[INFO] Ignorando colunas de clima (noclim): {CLIMATE_COLS}")
    drop_cols += leak_cols

    # Filtrar colunas de X
    X_cols = [c for c in num_cols if c not in drop_cols]

    if not X_cols:
        raise ValueError("Nenhuma feature numérica sobrou para X. Verificar seleção de colunas.")

    print(f"[INFO] Nº de features numéricas em X: {len(X_cols)}")
    print(f"[INFO] Algumas features: {X_cols[:10]}")

    X = df[X_cols].values
    y = df[target_col].values

    # Tratar NaNs em X
    mask = ~np.isnan(X).any(axis=1)
    if mask.sum() < len(X):
        print(f"[INFO] Removendo {len(X) - mask.sum()} linhas com NaN em X.")
        X = X[mask]
        y = y[mask]

    # Escalonar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X_cols

def compute_feature_importance(X, y, feature_names, random_state=42):
    # Modelo moderado, focado em ranking de importância
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
    )
    print("[INFO] Treinando XGBRegressor para importância de features...")
    model.fit(X, y)

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Ordenar e adicionar rank
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["rank"] = np.arange(1, len(imp_df) + 1)

    return imp_df

def main():
    scenarios = [
        # tag, target_col, delta_max_days, use_climate
        ("CP_D5_noclim",      "CP",            5, False),
        ("CP_D5_clim",        "CP",            5, True),
        ("CP_D7_noclim",      "CP",            7, False),
        ("CP_D7_clim",        "CP",            7, True),
        ("TDN_D5_noclim",     "TDN_based_ADF", 5, False),
        ("TDN_D5_clim",       "TDN_based_ADF", 5, True),
        ("TDN_D7_noclim",     "TDN_based_ADF", 7, False),
        ("TDN_D7_clim",       "TDN_based_ADF", 7, True),
    ]

    summary_rows = []

    for tag, target_col, dmax, use_clim in scenarios:
        try:
            X, y, feat_names = prepare_dataset(target_col, dmax, use_clim)
        except Exception as e:
            print(f"[AVISO] Pulando cenário {tag} por erro: {e}")
            continue

        imp_df = compute_feature_importance(X, y, feat_names)

        # Salvar CSV completo por cenário (todas as features)
        out_csv = OUT_DIR / f"feature_importance_xgb_{tag}.csv"
        imp_df.to_csv(out_csv, index=False)
        print(f"[OK] Importância de features ({tag}) salva em: {out_csv}")

        # Adicionar ao resumo global (todas as features)
        for _, row in imp_df.iterrows():
            summary_rows.append({
                "scenario": tag,
                "target_col": target_col,
                "delta_max_days": dmax,
                "use_climate": use_clim,
                "rank": int(row["rank"]),
                "feature": row["feature"],
                "importance": float(row["importance"]),
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_out = OUT_DIR / "feature_importance_xgb_all_scenarios.csv"
        summary_df.to_csv(summary_out, index=False)
        print(f"[OK] Resumo global (todas as features, todos os cenários) salvo em: {summary_out}")
    else:
        print("[AVISO] Nenhum cenário foi processado com sucesso; verificar erros acima.")

if __name__ == "__main__":
    main()
