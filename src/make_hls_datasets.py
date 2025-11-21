# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import pandas as pd
from pathlib import Path

RAW = "/workspace/data/data_raw/Complete_DataSet.csv"
OUTDIR = Path("/workspace/data/data_processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Heurística de colunas climáticas (pelo seu CSV)
CLIMATE_KEYS = [
    "TEMP", "RAD", "RAIN", "WIND", "EVAP", "PRES", "HUM", "TP_SFC",
    "Dew_Point", "Radiative_", "PPFD", "Longwave_Rad"
]

# Bandas/índices espectrais do seu CSV
SPECTRAL_COLS = [
    "B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12",
    "NDVI","NDWI","EVI","LAI","DVI","GCI","GEMI","SAVI"
]

def is_climate(col: str) -> bool:
    for k in CLIMATE_KEYS:
        if k in col:
            return True
    return False

def load_base() -> pd.DataFrame:
    df = pd.read_csv(RAW)

    # Normaliza nomes principais para o que seus scripts esperam
    rename_map = {
        "lat": "Latitude",
        "lon": "Longitude",
        "Sample": "SampleID"
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Se não existir SampleID, cria a partir de Sample/Sub-Sample
    if "SampleID" not in df.columns and {"Sample","Sub-Sample"}.issubset(df.columns):
        df["SampleID"] = df["Sample"].astype(str) + "_" + df["Sub-Sample"].astype(str)

    # Datas em datetime
    for c in ["Date","Satellite_Images_Dates"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df

def add_delta(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns and "Satellite_Images_Dates" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["Date"]) and pd.api.types.is_datetime64_any_dtype(df["Satellite_Images_Dates"]):
            df["Delta_days"] = (df["Date"] - df["Satellite_Images_Dates"]).abs().dt.days
            return df
    df["Delta_days"] = pd.NA
    return df

def split_cols(df: pd.DataFrame):
    all_cols = list(df.columns)
    climate_cols = [c for c in all_cols if is_climate(c)]
    spectral_cols = [c for c in SPECTRAL_COLS if c in all_cols]
    return spectral_cols, climate_cols

def common_cols(df: pd.DataFrame):
    base = ["SampleID","Date","Latitude","Longitude","CP","TDN_based_ADF","Delta_days"]
    return [c for c in base if c in df.columns]

def build_view(df: pd.DataFrame, spectral_cols, climate_cols, use_climate: bool):
    cols = common_cols(df) + spectral_cols + (climate_cols if use_climate else [])
    # unicidade + só colunas existentes
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    view = df[cols].copy()
    # limpeza mínima
    reqs = [c for c in ["Date","Latitude","Longitude"] if c in view.columns]
    if reqs:
        view = view.dropna(subset=reqs)
    return view

def write_csv(df: pd.DataFrame, name: str):
    out = OUTDIR / name
    df.to_csv(out, index=False)
    print(f">> {out} (linhas={len(df)}, cols={len(df.columns)})")

def main():
    df = load_base()
    df = add_delta(df)
    spectral_cols, climate_cols = split_cols(df)

    # RAW
    raw_noclim = build_view(df, spectral_cols, climate_cols, use_climate=False)
    raw_clim   = build_view(df, spectral_cols, climate_cols, use_climate=True)

    # D5 / D7 (se tivermos Delta_days)
    d5_noclim, d5_clim = raw_noclim.copy(), raw_clim.copy()
    d7_noclim, d7_clim = raw_noclim.copy(), raw_clim.copy()
    if "Delta_days" in df.columns:
        if "Delta_days" in d5_noclim.columns: d5_noclim = d5_noclim[d5_noclim["Delta_days"].le(5)]
        if "Delta_days" in d5_clim.columns:   d5_clim   = d5_clim[d5_clim["Delta_days"].le(5)]
        if "Delta_days" in d7_noclim.columns: d7_noclim = d7_noclim[d7_noclim["Delta_days"].le(7)]
        if "Delta_days" in d7_clim.columns:   d7_clim   = d7_clim[d7_clim["Delta_days"].le(7)]

    # salvar (os 6 novos “HLS-like” para comparação)
    write_csv(raw_noclim, "hls_RAW_noclim.csv")
    write_csv(raw_clim,   "hls_RAW_clim.csv")
    write_csv(d5_noclim,  "hls_D5_noclim.csv")
    write_csv(d5_clim,    "hls_D5_clim.csv")
    write_csv(d7_noclim,  "hls_D7_noclim.csv")
    write_csv(d7_clim,    "hls_D7_clim.csv")

if __name__ == "__main__":
    main()
