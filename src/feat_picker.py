# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

# -*- coding: utf-8 -*-
import re
import pandas as pd

SPECTRAL = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12",
            "NDVI","NDWI","EVI","LAI","DVI","GCI","GEMI","SAVI"]

NOT_FEATS = {
    "Date","Satellite_Images_Dates","DeltaDays","heldout_date","fold_date","date",
    "ID","Id","id","fid","FID","Unnamed: 0","__index__","sample_id",
    "lat","lon","Lat","Lon","Latitude","Longitude",
    "CP","TDN_based_ADF","TDN_based_NDF"
}

CLIMATE_RE = re.compile(
    r"(TEMP|T2M|TAVG|TMAX|TMIN|PREC|PRCP|RAIN|RR|CHIRPS|ERA5|RH|DEW|WIND|WS|WD|"
    r"PRESS|SLP|VPD|EVAP|ET0|SOIL|SM|HUM)", re.I
)

def pick_features(df: pd.DataFrame, target_col: str) -> list[str]:
    not_feats = set(NOT_FEATS) | {target_col}
    spec = [c for c in SPECTRAL if c in df.columns]
    num = [c for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c])
           and c not in spec and c not in not_feats]
    clim = [c for c in num if CLIMATE_RE.search(c or "")]
    return spec + clim
