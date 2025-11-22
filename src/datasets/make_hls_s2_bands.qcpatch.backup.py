# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python3
"""
make_hls_s2_bands.py — Gera datasets HLS-S2 (bands-only) no modelo RAW, D5 e D7

Objetivo: recriar os datasets do UFMS usando **apenas bandas** do HLS-S2 (HLSS30),
mesmo buffer, seguindo as datas/labels do `Complete_DataSet.csv`.

Modos:
  - --mode raw : cena mais próxima da data de laboratório
  - --mode d5  : mediana das cenas válidas na janela ±5 dias
  - --mode d7  : mediana das cenas válidas na janela ±7 dias

Saídas (automáticas por modo):
  - RAW → /workspace/data/data_processed/Complete_DataSet_HLS_S2.csv
  - D5  → /workspace/data/data_processed/Complete_DataSet_HLS_S2_d5.csv
  - D7  → /workspace/data/data_processed/Complete_DataSet_HLS_S2_d7.csv
  - QC  → /workspace/reports/progress/qc_HLS_S2_{RAW|D5|D7}.{md,csv}
  - (opcional) Parquet espelhado via --out-parquet

Regras (preservar linhas, sem vazamentos):
  - QA permissivo: remover cloud (4) e shadow (2) via Fmask; sem dilatação
  - RAW: cena mais próxima; empate → menor nuvem no buffer, depois maior n_valid
  - D5/D7: agregar **apenas** dentro da janela; estatística = **mediana**
  - Sem clima/índices/ângulos nesta rodada (bands-only)

Dependências (instale no contêiner):
  pip install planetary-computer pystac-client rasterio shapely pyproj pandas numpy tqdm

Autor: PROJETO MESTRADO (2025-10-26)
"""
from __future__ import annotations
import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from shapely.geometry import Point, mapping
from shapely.ops import transform as shp_transform
import pyproj
import rasterio
from rasterio.features import geometry_window

import pystac_client
import planetary_computer as pc

# ----------------------------
# Configuração de bandas HLS
# ----------------------------
BAND_MAP = {
    "B02": "spec_blue",
    "B03": "spec_green",
    "B04": "spec_red",
    "B05": "spec_rededge1",
    "B06": "spec_rededge2",
    "B07": "spec_rededge3",
    "B08": "spec_nir",
    "B11": "spec_swir1",
    "B12": "spec_swir2",
}
FMASK_ASSET = "Fmask"  # 0 clear, 1 water, 2 shadow, 3 snow/ice, 4 cloud, 255 fill
CLEAR_CODES = {0, 1}
MASKED_CODES = {2, 4}

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
HLS_COLLECTION = "hls2-s30"
# ----------------------------
# Geo utilitários
# ----------------------------
@dataclass
class SceneQC:
    item_id: str
    scene_datetime: pd.Timestamp
    abs_days: int
    cloud_frac_buffer: float
    n_valid_total: int


def _utm_crs_for(lon: float, lat: float) -> pyproj.CRS:
    info = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon, south_lat_degree=lat, east_lon_degree=lon, north_lat_degree=lat
        ),
    )[0]
    return pyproj.CRS.from_user_input(info.code)


def reproject_point_buffer(lon: float, lat: float, buffer_m: float, dst_crs: str) -> Tuple[dict, pyproj.CRS]:
    wgs84 = pyproj.CRS("EPSG:4326")
    dst = pyproj.CRS(dst_crs)
    utm = _utm_crs_for(lon, lat)
    to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    to_dst = pyproj.Transformer.from_crs(utm, dst, always_xy=True).transform
    circle = shp_transform(to_utm, Point(lon, lat)).buffer(buffer_m)
    circle_dst = shp_transform(to_dst, circle)
    return mapping(circle_dst), dst


def read_fmask_window(fmask_href: str, geom: dict) -> Tuple[np.ndarray, float, int]:
    with rasterio.open(fmask_href) as src:
        window = geometry_window(src, [geom], pixel_precision=3)
        fm = src.read(1, window=window)
        valid_fill = (fm != 255)
        use = valid_fill & np.isin(fm, list(CLEAR_CODES))
        cloudish = np.isin(fm, list(MASKED_CODES))
        total = int(valid_fill.sum())
        cloud_frac = float(cloudish.sum()) / float(total) if total > 0 else 1.0
        return use, cloud_frac, total


def read_band_mean(href: str, geom: dict, use_mask: np.ndarray, prefer_median: bool = False) -> Tuple[Optional[float], int]:
    with rasterio.open(href) as src:
        window = geometry_window(src, [geom], pixel_precision=3)
        arr = src.read(1, window=window).astype(np.float32)
        valid = np.isfinite(arr) & use_mask
        if valid.sum() == 0:
            return None, 0
        val = float(np.median(arr[valid]) if prefer_median else np.mean(arr[valid]))
        return val, int(valid.sum())
# ----------------------------
# STAC helpers
# ----------------------------
def stac_search_items(lon: float, lat: float, date: pd.Timestamp, days_window: int, limit: int):
    client = pystac_client.Client.open(STAC_URL)
    dt_min = (date - pd.Timedelta(days=days_window)).strftime("%Y-%m-%dT00:00:00Z")
    dt_max = (date + pd.Timedelta(days=days_window)).strftime("%Y-%m-%dT23:59:59Z")
    search = client.search(
        collections=[HLS_COLLECTION],
        intersects={"type": "Point", "coordinates": [float(lon), float(lat)]},
        datetime=f"{dt_min}/{dt_max}",
        limit=limit,
    )
    return list(search.get_items())


def pick_best_raw(items, target_date: pd.Timestamp, lon: float, lat: float, buffer_m: float):
    """Escolhe a melhor cena para RAW: menor |Δ dias|; desempate por nuvem e n_valid."""
    best = None
    best_assets = None
    best_geom = None
    for it in items:
        dt = pd.to_datetime(it.datetime).tz_localize(None)
        abs_days = int(abs((dt.date() - target_date.date()).days))
        signed = pc.sign(it)  # assina URLs para leitura
        if FMASK_ASSET not in signed.assets:
            continue
        fmask_href = signed.assets[FMASK_ASSET].href
        with rasterio.open(fmask_href) as src_fm:
            geom, _ = reproject_point_buffer(lon, lat, buffer_m, src_fm.crs.to_string())
        use_mask, cloud_frac, n_valid_total = read_fmask_window(fmask_href, geom)
        qc = SceneQC(
            item_id=it.id,
            scene_datetime=dt,
            abs_days=abs_days,
            cloud_frac_buffer=cloud_frac,
            n_valid_total=n_valid_total,
        )
        if best is None:
            best, best_assets, best_geom = qc, {k: v.href for k, v in signed.assets.items()}, geom
        else:
            swap = False
            if qc.abs_days < best.abs_days:
                swap = True
            elif qc.abs_days == best.abs_days:
                if qc.cloud_frac_buffer < best.cloud_frac_buffer:
                    swap = True
                elif math.isclose(qc.cloud_frac_buffer, best.cloud_frac_buffer, rel_tol=1e-6):
                    if qc.n_valid_total > best.n_valid_total:
                        swap = True
            if swap:
                best, best_assets, best_geom = qc, {k: v.href for k, v in signed.assets.items()}, geom
    return best, best_assets, best_geom


def aggregate_window(items, lon: float, lat: float, buffer_m: float, prefer_median: bool = True):
    """Agrega medianas por banda em todas as cenas válidas na janela (para D5/D7)."""
    per_band_vals = {v: [] for v in BAND_MAP.values()}
    qcs: List[SceneQC] = []
    for it in items:
        signed = pc.sign(it)
        if FMASK_ASSET not in signed.assets:
            continue
        fmask_href = signed.assets[FMASK_ASSET].href
        with rasterio.open(fmask_href) as src_fm:
            geom, _ = reproject_point_buffer(lon, lat, buffer_m, src_fm.crs.to_string())
        use_mask, cloud_frac, n_valid_total = read_fmask_window(fmask_href, geom)
        if n_valid_total == 0 or (use_mask.sum() == 0):
            continue
        dt = pd.to_datetime(it.datetime).tz_localize(None)
        qcs.append(SceneQC(item_id=it.id, scene_datetime=dt, abs_days=0, cloud_frac_buffer=cloud_frac, n_valid_total=n_valid_total))
        for b, out_name in BAND_MAP.items():
            a = signed.assets.get(b)
            if a is None:
                continue
            val, nvalid = read_band_mean(a.href, geom, use_mask, prefer_median=prefer_median)
            if val is not None:
                per_band_vals[out_name].append(val)
    out = {}
    for out_name, vals in per_band_vals.items():
        out[out_name] = float(np.median(vals) if (prefer_median and len(vals)>0) else (np.mean(vals) if len(vals)>0 else np.nan))
    return out, qcs
# ----------------------------
# Pipeline por linha (RAW, D5, D7)
# ----------------------------
def process_row_mode(row: pd.Series, mode: str, buffer_m: float, limit: int):
    sid = row.get("SampleID", row.get("Id", row.get("ID", row.get("id"))))
    date = pd.to_datetime(row["Date"])  # data de laboratório
    lon = float(row.get("lon", row.get("Lon", row.get("Longitude"))))
    lat = float(row.get("lat", row.get("Lat", row.get("Latitude"))))

    # busca ampla para achar candidatos perto da data
    base_window = 15
    items = stac_search_items(lon, lat, date, days_window=base_window, limit=limit)
    if not items:
        return None, {"SampleID": sid, "mode": mode, "has_item": 0}

    if mode == "raw":
        best, assets, geom = pick_best_raw(items, date, lon, lat, buffer_m)
        if best is None or assets is None or geom is None:
            return None, {"SampleID": sid, "mode": mode, "has_item": 0}
        # máscara Fmask para a cena escolhida
        fmask_href = assets.get(FMASK_ASSET)
        if fmask_href is None:
            return None, {"SampleID": sid, "mode": mode, "has_item": 1, "scene_id": best.item_id}
        use_mask, cloud_frac, n_valid_total = read_fmask_window(fmask_href, geom)
        out = {}
        n_valid_any = 0
        for b, out_name in BAND_MAP.items():
            a = assets.get(b)
            if a is None:
                out[out_name] = np.nan
                continue
            val, nvalid = read_band_mean(a, geom, use_mask, prefer_median=False)
            if val is None:
                out[out_name] = np.nan
            else:
                out[out_name] = val
                n_valid_any += nvalid
        used = int(n_valid_any > 0)
        qc = {
            "SampleID": sid,
            "mode": mode,
            "has_item": 1,
            "scene_id": best.item_id,
            "scene_datetime": best.scene_datetime.isoformat(),
            "abs_days": best.abs_days,
            "cloud_frac_buffer": cloud_frac,
            "n_valid_total": n_valid_total,
            "used": used,
        }
        return (out if used else None), qc

    # modos D5 / D7
    win = 5 if mode == "d5" else 7
    # filtrar itens que caem dentro da janela fina ±win
    items_win = [it for it in items if abs((pd.to_datetime(it.datetime).tz_localize(None).date() - date.date()).days) <= win]
    if not items_win:
        return None, {"SampleID": sid, "mode": mode, "has_item": 0, "in_window": 0}
    out, qcs = aggregate_window(items_win, lon, lat, buffer_m, prefer_median=True)
    # se todas bandas virarem NA, considerar inválido
    if all(pd.isna(v) for v in out.values()):
        return None, {"SampleID": sid, "mode": mode, "has_item": 1, "in_window": 1, "used": 0}
    qc = {
        "SampleID": sid,
        "mode": mode,
        "has_item": 1,
        "in_window": 1,
        "n_scenes_window": len(qcs),
        "scene_datetimes": ",".join(sorted({q.scene_datetime.isoformat() for q in qcs})),
        "cloud_frac_median": float(np.median([q.cloud_frac_buffer for q in qcs])) if qcs else None,
    }
    return out, qc
# ----------------------------
# Main CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Gera datasets HLS-S2 bands-only: RAW, D5, D7")
    ap.add_argument("--mode", choices=["raw", "d5", "d7"], required=True)
    ap.add_argument("--csv", default="/workspace/data/data_raw/Complete_DataSet.csv")
    ap.add_argument("--buffer-m", type=float, default=30.0)
    ap.add_argument("--out-csv", default=None, help="Caminho CSV de saída (se não dado, usa o padrão por modo)")
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--qc-csv", default=None)
    ap.add_argument("--qc-md", default=None)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    # destinos padrão por modo
    default_out = {
        "raw": "/workspace/data/data_processed/Complete_DataSet_HLS_S2.csv",
        "d5":  "/workspace/data/data_processed/Complete_DataSet_HLS_S2_d5.csv",
        "d7":  "/workspace/data/data_processed/Complete_DataSet_HLS_S2_d7.csv",
    }
    default_qc_csv = {
        "raw": "/workspace/reports/progress/qc_HLS_S2_RAW.csv",
        "d5":  "/workspace/reports/progress/qc_HLS_S2_D5.csv",
        "d7":  "/workspace/reports/progress/qc_HLS_S2_D7.csv",
    }
    default_qc_md = {
        "raw": "/workspace/reports/progress/qc_HLS_S2_RAW.md",
        "d5":  "/workspace/reports/progress/qc_HLS_S2_D5.md",
        "d7":  "/workspace/reports/progress/qc_HLS_S2_D7.md",
    }

    out_csv = args.out_csv or default_out[args.mode]
    qc_csv = args.qc_csv or default_qc_csv[args.mode]
    qc_md  = args.qc_md  or default_qc_md[args.mode]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(qc_csv), exist_ok=True)
    os.makedirs(os.path.dirname(qc_md), exist_ok=True)

    df = pd.read_csv(args.csv)
    # Normalizações mínimas
    if "Date" not in df.columns:
        raise SystemExit("CSV UFMS precisa ter a coluna 'Date'")
    # Coordenadas
    lon_col = next((c for c in ["lon","Lon","LON","Longitude","LONGITUDE","long"] if c in df.columns), None)
    lat_col = next((c for c in ["lat","Lat","LAT","Latitude","LATITUDE"] if c in df.columns), None)
    if not lon_col or not lat_col:
        raise SystemExit("CSV UFMS precisa ter colunas de coordenadas (lon/lat)")
    if "SampleID" not in df.columns:
        sid_cand = next((c for c in ["Id","ID","id"] if c in df.columns), None)
        if sid_cand:
            df = df.rename(columns={sid_cand: "SampleID"})
        else:
            df.insert(0, "SampleID", np.arange(1, len(df)+1))

    out_rows = []
    qc_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Build {args.mode.upper()}"):
        try:
            feats, qc = process_row_mode(row, args.mode, args.buffer_m, args.limit)
        except Exception as e:
            feats, qc = None, {"SampleID": row.get("SampleID", None), "mode": args.mode, "error": str(e)}
        qc_rows.append(qc)
        if feats is None:
            continue
        base = {"SampleID": row["SampleID"], "Date": row["Date"]}
        for t in ["CP", "TDN_based_ADF"]:
            if t in df.columns:
                base[t] = row[t]
        out_rows.append({**base, **feats})

    if len(out_rows) == 0:
        raise SystemExit("Sem linhas válidas. Revise máscara/CRS/buffer.")

    df_out = pd.DataFrame(out_rows)
    # Ordenação das colunas
    cols = ["SampleID", "Date"]
    for t in ["CP", "TDN_based_ADF"]:
        if t in df_out.columns:
            cols.append(t)
    spec_cols = [name for name in ["spec_blue","spec_green","spec_red","spec_rededge1","spec_rededge2","spec_rededge3","spec_nir","spec_swir1","spec_swir2"] if name in df_out.columns]
    cols.extend(spec_cols)
    df_out = df_out[cols]

    df_out.to_csv(out_csv, index=False)
    if args.out_parquet:
        df_out.to_parquet(args.out_parquet, index=False)

    # QC
    df_qc = pd.DataFrame(qc_rows)
    df_qc.to_csv(qc_csv, index=False)

    # Resumo QC
    n_total = len(df)
    n_used = len(df_out)
    used_pct = 100.0 * n_used / n_total if n_total else 0.0
    with open(qc_md, "w", encoding="utf-8") as f:
        f.write(f"# QC — HLS_S2_{args.mode.upper()}_BANDS\n\n")
        f.write(f"Total UFMS: {n_total}\n")
        f.write(f"Com bandas válidas: {n_used} ({used_pct:.1f}%)\n\n")
        if args.mode == "raw":
            abs_days = pd.to_numeric(df_qc.get("abs_days"), errors="coerce")
            if abs_days is not None:
                desc = abs_days.describe(percentiles=[0.5, 0.9, 0.95]).to_dict()
                f.write("## |Δ dias| RAW\n\n")
                for k in ["count","mean","50%","90%","95%","max"]:
                    if k in desc and pd.notna(desc[k]):
                        f.write(f"{k}: {desc[k]}\n")
        else:
            if "n_scenes_window" in df_qc.columns:
                vals = pd.to_numeric(df_qc["n_scenes_window"], errors="coerce")
                desc = vals.describe(percentiles=[0.25,0.5,0.75,0.9]).to_dict()
                f.write("## Cenas por janela\n\n")
                for k in ["count","50%","75%","90%","max"]:
                    if k in desc and pd.notna(desc[k]):
                        f.write(f"{k}: {desc[k]}\n")
        f.write("\n## Observações\n\n- QA: Fmask (clear/water mantidos; cloud/shadow removidos).\n- Buffer: raio em metros informado via --buffer-m.\n- Modos D5/D7: mediana por banda dentro da janela.\n")

    print(f"OK {args.mode.upper()}: {out_csv}")
    print(f"QC: {qc_md} | {qc_csv}")
if __name__ == "__main__":
    main()

# --- override: Fmask como BITMASK (HLSS30) ---
# Referência de bits (HLSS30, Earth Engine docs):
#  bit1: cloud (1=sim), bit3: cloud shadow (1=sim), bit5: water (1=sim)
#  bit2 adjacente, bit4 neve/gelo – vamos ser permissivos e NÃO excluir por eles.
def read_fmask_window(fmask_href: str, geom: dict):
    import numpy as np, rasterio
    from rasterio.features import geometry_window
    with rasterio.open(fmask_href) as src:
        window = geometry_window(src, [geom], pixel_precision=3)
        fm = src.read(1, window=window).astype(np.uint16)
        # nodata robusto
        nd = src.nodata
        valid = np.ones_like(fm, dtype=bool) if nd is None else (fm != nd)
        # bits (EE doc HLSS30)
        cloud  = ((fm >> 1) & 1).astype(bool)
        shadow = ((fm >> 3) & 1).astype(bool)
        # água é permitida; adj (bit2) e neve (bit4) não excluímos por ora
        cloudish = cloud | shadow
        use = valid & (~cloudish)
        total = int(valid.sum())
        cloud_frac = float(cloudish.sum()) / float(total) if total > 0 else 1.0
        return use, cloud_frac, total

# --- override: Fmask como BITMASK (HLS2-S30, Planetary Computer) ---
# Bits (conforme documentação HLS no EE/PC):
#   bit1: cloud (1=sim)
#   bit3: cloud shadow (1=sim)
#   bit5: water (1=sim)  -> permitido
# (bits adjacentes/neve não excluímos por ora; só nuvem/sombra)
def read_fmask_window(fmask_href: str, geom: dict):
    import numpy as np, rasterio
    from rasterio.features import geometry_window
    with rasterio.open(fmask_href) as src:
        window = geometry_window(src, [geom], pixel_precision=3)
        fm = src.read(1, window=window).astype(np.uint16)
        nd = src.nodata
        valid = np.ones_like(fm, dtype=bool) if nd is None else (fm != nd)
        cloud  = ((fm >> 1) & 1).astype(bool)   # bit1
        shadow = ((fm >> 3) & 1).astype(bool)   # bit3
        cloudish = cloud | shadow
        use = valid & (~cloudish)               # água (bit5) é mantida
        total = int(valid.sum())
        cloud_frac = float(cloudish.sum()) / float(total) if total > 0 else 1.0
        return use, cloud_frac, total
