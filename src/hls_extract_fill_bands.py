# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import os, re, math, json
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from pyproj import Transformer
import requests
from urllib.parse import urlencode

DP       = "/workspace/data/data_processed"
MANIFEST = "/workspace/reports/progress/HLS_S30_manifest.csv"
IN_CSV   = f"{DP}/complete_S2_allgeom_HLS.csv"   # esqueleto
OUT_CSV  = f"{DP}/complete_S2_allgeom_HLS.csv"   # sobrescreve
CACHE    = "/workspace/cache/HLS_bands"
os.makedirs(CACHE, exist_ok=True)

# None = tudo; ou defina via env HLS_LIMIT
LIMIT = (lambda v: None if v in (None,"",) else int(v))(os.environ.get("HLS_LIMIT"))

# Bandas HLSS30 ~ Sentinel-2
HLS_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
def short_name(b):  # B01->B1, B02->B2, B8A->B8A
    m = re.fullmatch(r"B0?(\d+)", b);  return f"B{int(m.group(1))}" if m else b

def cmr_links_for_granule(concept_id:str):
    url = "https://cmr.earthdata.nasa.gov/search/granules.json?" + urlencode({"concept_id":concept_id})
    r = requests.get(url, timeout=60); r.raise_for_status()
    js = r.json()
    entry = (js.get("feed") or {}).get("entry") or []
    return [L.get("href") for L in entry[0].get("links",[]) if isinstance(L,dict) and L.get("href")] if entry else []

def get_band_url(row, band:str):
    base_url = str(row.get("download_url") or "")
    if base_url.endswith(".B02.tif"):
        root = base_url.rsplit(".B02.tif",1)[0]
        return f"{root}.{band}.tif"
    # fallback via CMR
    links = cmr_links_for_granule(str(row["granule_id"]))
    for u in links:
        if isinstance(u,str) and u.endswith(f".{band}.tif"):
            return u
    return None

def ensure_local(url:str):
    fn = os.path.join(CACHE, os.path.basename(url))
    if not os.path.exists(fn):
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(fn,"wb") as f:
                for chunk in r.iter_content(1024*512):
                    if chunk: f.write(chunk)
    return fn

def stats_window(arr):
    v = arr[np.isfinite(arr)]
    if v.size==0: return np.nan, np.nan
    med = float(np.median(v))
    iqr = float(np.subtract(*np.percentile(v, [75,25])))
    return med, iqr

def extract_stats_for_point(tif_path, lat, lon):
    # abre e projeta
    with rasterio.open(tif_path) as ds:
        # reprojeta coord WGS84 -> ds.crs
        transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        # pixel center
        px, py = ~ds.transform * (x, y)
        px_i, py_i = int(round(px)), int(round(py))

        # 2DAP5x5: janela 5x5 centrada
        w = Window(px_i-2, py_i-2, 5, 5).intersection(Window(0,0, ds.width, ds.height))
        a_5x5 = ds.read(1, window=w).astype("float32")
        med5, iqr5 = stats_window(a_5x5)

        # BUF30m: raio em pixels (aprox. usa escala X do transform, em metros)
        # HLS em UTM -> transform.a ~ pixel_size (m), ~30m/10m=3 px
        # faremos um quadrado 9×9 com máscara circular de raio ≈3 px
        radius_px = max(1, int(round(30.0 / abs(ds.transform.a))))
        w2 = Window(px_i-radius_px, py_i-radius_px, 2*radius_px+1, 2*radius_px+1).intersection(Window(0,0, ds.width, ds.height))
        a_buf = ds.read(1, window=w2).astype("float32")
        # máscara circular
        h, w_ = a_buf.shape
        yy, xx = np.ogrid[:h, :w_]
        cy, cx = h//2, w_//2
        mask = (xx-cx)*(xx-cx) + (yy-cy)*(yy-cy) <= radius_px*radius_px
        a_buf = np.where(mask, a_buf, np.nan)
        medb, iqrb = stats_window(a_buf)

        return (med5, iqr5, medb, iqrb)

def main():
    df = pd.read_csv(IN_CSV)  # esqueleto com nomes/ordem finais
    man = pd.read_csv(MANIFEST)

    keycols = ["Sample","Sub_Sample","Date_std"]
    # merge para ter granule por linha
    m = pd.merge(df[keycols+["Lat","Lon"]], man[keycols+["granule_id","download_url"]], on=keycols, how="left")
    assert m["granule_id"].notna().all(), "Manifest está incompleto para alguma linha."

    # quais colunas preencher
    geom_suffixes = ["__2DAP5x5_MED","__2DAP5x5_IQR","__BUF30m_MED","__BUF30m_IQR"]
    fill_cols = [f"B{n}{suf}" for n in list(range(1,8))+[8,"8A",9,11,12] for suf in geom_suffixes]  # sem B10
    # vamos processar em iterator para economizar memória
    to_process = m.itertuples(index=False)
    if LIMIT: to_process = (r for i,r in enumerate(to_process) if i < LIMIT)

    # map: (row_index_in_df -> dict col->val)
    updates = {}
    for i, row in enumerate(to_process, 1):
        lat, lon = float(row.Lat), float(row.Lon)
        for band in HLS_BANDS:
            sname = short_name(band)  # "B1", "B2", ...
            if sname == "B10":  # não existe em HLSS30
                continue
            url = get_band_url(row._asdict(), band)
            if not url:  # não achou a banda (incomum); pula
                continue
            tif = ensure_local(url)
            try:
                med5, iqr5, medb, iqrb = extract_stats_for_point(tif, lat, lon)
            except Exception:
                med5 = iqr5 = medb = iqrb = math.nan
            # aplica nos nomes de coluna
            rec = updates.setdefault((row.Sample, row.Sub_Sample, row.Date_std), {})
            rec[f"{sname}__2DAP5x5_MED"] = med5
            rec[f"{sname}__2DAP5x5_IQR"] = iqr5
            rec[f"{sname}__BUF30m_MED"]  = medb
            rec[f"{sname}__BUF30m_IQR"]  = iqrb

        if i % 20 == 0:
            print(f"[{i}] linhas processadas…", flush=True)

    # aplica updates no df (sem mexer na ordem)
    df_up = df.copy()
    df_up.set_index(keycols, inplace=True)
    for k, vals in updates.items():
        for c, v in vals.items():
            if c in df_up.columns:
                df_up.at[k, c] = v
    df_up.reset_index(inplace=True)
    df_up.to_csv(OUT_CSV, index=False)
    print(f"[DONE] atualizou {OUT_CSV} com {len(updates)} linhas (LIMIT={LIMIT}).")

if __name__ == "__main__":
    main()
