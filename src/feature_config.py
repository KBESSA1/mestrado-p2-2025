# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


CSV_PATH_DEFAULT = "/workspace/data/data_raw/Complete_DataSet.csv"

# Colunas de clima (para poder tirar no modo "noclim")
CLIMATE_COLS = [
    "TEMP_MAX",
    "TEMP_MIN",
    "RAIN",
    "RAD_SOL",
    "Radiative_Dif_AVG",
    "Radiative_Direct_AVG",
    "Longwave_Rad_AVG",
    "TP_SFC_AVG",
    "PRES_ATM",
    "EVAPOT",
    "WIND_SPD",
    "Wind_Dir",
    "HUM_REL",
    "PPFD",
]

# === Feature set para CP (derivado da análise de importância XGB) ===
CP_FEATURES = [
    "Animals",
    "DM",
    "Biomass",
    "ADF",
    "TDN_based_NDF",
    "TDN_based_ADF",
    "RAD_SOL",
    "Radiative_Dif_AVG",
    "Radiative_Direct_AVG",
    "Longwave_Rad_AVG",
    "TP_SFC_AVG",
    "TEMP_MIN",
    "WIND_SPD",
    "Wind_Dir",
    "EVAPOT",
    "PPFD",
    "DOY",
    "GEMI",
    "DVI",
    "B6",
    "B7",
    "B9",
]

# === Feature set base para TDN_based_ADF (sem leaks) ===
TDN_FEATURES_BASE = [
    "Animals",
    "DM",
    "Biomass",
    "NDF",
    "Radiative_Dif_AVG",
    "PRES_ATM",
    "Wind_Dir",
    "RAD_SOL",
    "TP_SFC_AVG",
    "TEMP_MAX",
    "TEMP_MIN",
    "EVAPOT",
    "Radiative_Direct_AVG",
    "HUM_REL",
    "PPFD",
    "B1",
    "B6",
    "B8",
    "B11",
    "B12",
    "GCI",
    "SAVI",
    "LAI",
]


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Coluna de data '{col}' não encontrada no CSV.")
    return pd.to_datetime(df[col], errors="coerce")


def prepare_xy(
    csv_path: str = CSV_PATH_DEFAULT,
    target: str = "CP",
    delta_max_days: int = 5,
    use_climate: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Lê o Complete_DataSet, aplica filtro de Δ dias (D5/D7),
    seleciona as features de acordo com o alvo (CP ou TDN_based_ADF)
    e retorna:
        X (DataFrame numérico),
        y (np.ndarray),
        df_filt (DataFrame filtrado, incluindo Date para o LODO).
    """

    print(f"[INFO] Lendo CSV de {csv_path}")
    df = pd.read_csv(csv_path)

    # Garantir colunas de data
    df["Date"] = _ensure_datetime(df, "Date")
    df["Satellite_Images_Dates"] = _ensure_datetime(df, "Satellite_Images_Dates")

    # Remover nulos em datas e alvo
    before = len(df)
    df = df.dropna(subset=["Date", "Satellite_Images_Dates", target]).copy()
    after = len(df)
    print(
        f"[INFO] Linhas após remover nulos em datas e alvo ({target}): "
        f"{after} (removidas {before - after})"
    )

    # Garantir coluna DOY (caso não exista)
    if "DOY" not in df.columns:
        df["DOY"] = df["Date"].dt.dayofyear

    # Calcular Delta_days
    print("[INFO] Calculando Delta_days...")
    df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).dt.days.abs()

    # Filtro D5 / D7
    df = df[df["Delta_days"] <= delta_max_days].copy()
    print(f"[INFO] Nº de amostras após filtro D{delta_max_days}: {len(df)}")

    # Escolher feature set base conforme o alvo
    if target == "CP":
        feature_list = list(CP_FEATURES)
    elif target == "TDN_based_ADF":
        feature_list = list(TDN_FEATURES_BASE)
    else:
        raise ValueError(f"Alvo não suportado em prepare_xy: {target}")

    # Remover leaks no caso de TDN_based_ADF (garantia extra)
    if target == "TDN_based_ADF":
        leak_cols = ["ADF", "TDN_based_NDF", "CP"]
        feature_list = [f for f in feature_list if f not in leak_cols]

    # Se for cenário "noclim", tirar todas as colunas de clima
    if not use_climate:
        print(f"[INFO] Ignorando colunas de clima (noclim): {CLIMATE_COLS}")
        feature_list = [f for f in feature_list if f not in CLIMATE_COLS]

    # Manter apenas as features que realmente existem no DF
    feature_list = [f for f in feature_list if f in df.columns]

    print(f"[INFO] Nº de features numéricas candidatas em X: {len(feature_list)}")
    if len(feature_list) == 0:
        raise ValueError("Nenhuma feature restante após filtros de clima/leak.")

    print(f"[INFO] Algumas features: {feature_list[:10]}")

    # X: somente colunas numéricas entre as escolhidas
    X = df[feature_list].select_dtypes(include=[np.number]).copy()
    y = df[target].astype(float).to_numpy()

    # Por segurança, alinhar X e y
    if len(X) != len(y):
        raise RuntimeError("Tamanhos de X e y não batem após filtragem.")

    return X, y, df
