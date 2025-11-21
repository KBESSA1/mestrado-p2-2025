# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import os, re, json, math, time
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer
import requests
from urllib.parse import urlencode

# --- Paths ---
DP        = "/workspace/data/data_processed"
IN_CSV    = f"{DP}/complete_S2_allgeom_HLS.csv"           # esqueleto (S2 layout)
OUT_CSV   = f"{DP}/complete_S2_allgeom_HLS.csv"           # sobrescreve preenchendo
MANIFEST  = "/workspace/reports/progress/HLS_S30_manifest.csv"
CACHE_DIR = "/workspace/cache/HLS_bands"
LINKCACHE = "/workspace/reports/progress/HLS_linkcache.json"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Env / params ---
LIMIT         = (lambda v: None if not v else int(v))(os.environ.get("HLS_LIMIT"))
FLUSH_EVERY   = int(os.environ.get("HLS_FLUSH_EVERY", "20"))
VERBOSE       = bool(int(os.environ.get("HLS_VERBOSE", "0")))
ONLY_MIN_SET  = bool(int(os.environ.get("HLS_BANDS_MIN", "0")))  # 1 => só B02,B04,B08 primeiro
SESSION       = requests.Session()
SESSION.headers.update({"User-Agent":"UFMS-HLS-Extractor/0.2"})

# HLSS30 bands ~ Sentinel-2 (sem B10):
ALL_BANDS = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]
MIN_BANDS = ["B02","B04","B08"]
BANDS     = MIN_BANDS if ONLY_MIN_SET else ALL_BANDS

def short_name(b):  # B01->B1, B02->B2, B8A->B8A
    m = re.fullmatch(r"B0?(\d+)", b)
    return f"B{int(m.group(1))}" if m else b

def _load_linkcache():
    if os.path.exists(LINKCACHE):
        try:
            with open(LINKCACHE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_linkcache(cache):
    tmp = LINKCACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, LINKCACHE)

def cmr_links_for_granule(concept_id: str):
    # 1 chamada por granule e fica em cache
    params = {"concept_id": concept_id}
    url = "https://cmr.earthdata.nasa.gov/search/granules.json?" + urlencode(params)
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    entry = (js.get("feed") or {}).get("entry") or []
    if not entry:
        return []
    return [L.get("href") for L in entry[0].get("links", []) if isinstance(L, dict) and L.get("href")]

def band_url_for_row(row, band, linkcache):
    """
    row: pandas Series (com granule_id, download_url)
    band: 'B02', 'B04', ...
    Retorna URL .tif da banda.
    """
    base_url = str(row.get("download_url") or "")
    if base_url.endswith(".B02.tif"):
        root = base_url.rsplit(".B02.tif", 1)[0]
        return f"{root}.{band}.tif"

    gid = str(row["granule_id"])
    if gid in linkcache:
        root = linkcache[gid]
        return f"{root}.{band}.tif"

    # Sem base .B02 na planilha -> descobrir via CMR (1x) e cachear
    links = cmr_links_for_granule(gid)
    # procure um .B02 e derive a raiz
    b02 = [u for u in links if isinstance(u, str) and u.endswith(".B02.tif")]
    if not b02:
        # fallback: tente achar qualquer .B..tif e mapear por regex
        anyb = [u for u in links if isinstance(u, str) and re.search(r"\.B\d{2}\.tif$", u)]
        if not anyb:
            raise RuntimeError(f"Nenhuma URL .Bxx.tif encontrada para {gid}")
        m = re.search(r"^(.*)\.B\d{2}\.tif$", anyb[0])
    else:
        m = re.match(r"^(.*)\.B02\.tif$", b02[0])
    if not m:
        raise RuntimeError(f"Não foi possível inferir raiz para {gid}")
    root = m.group(1)
    linkcache[gid] = root
    _save_linkcache(linkcache)
    return f"{root}.{band}.tif"

def cache_path(url: str):
    return os.path.join(CACHE_DIR, os.path.basename(url))

def ensure_download(url: str):
    dst = cache_path(url)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    with SESSION.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dst + ".part", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*512):
                if chunk: f.write(chunk)
    os.replace(dst + ".part", dst)
    return dst

def med_iqr(vals: np.ndarray):
    # ignora NaNs
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return np.nan, np.nan
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    return med, iqr

def stats_on_geoms(ds, lon, lat):
    """
    Retorna dict:
      {'2DAP5x5': {'MED':..., 'IQR':...}, 'BUF30m': {'MED':..., 'IQR':...}}
    """
    # reprojeção
    to_img = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True).transform
    x, y = to_img(lon, lat)
    row, col = ds.index(x, y)

    # 2DAP5x5 = janela 5x5 centrada
    w = Window(col_off=max(col-2,0), row_off=max(row-2,0), width=5, height=5)
    arr = ds.read(1, window=w, boundless=True, masked=True).astype("float32")
    med5, iqr5 = med_iqr(arr.filled(np.nan))

    # BUF30m = círculo ~30m de raio (converte para pixels)
    # pixel size aproximado:
    res_x = abs(ds.transform.a)
    res_y = abs(ds.transform.e)
    pix = max(res_x, res_y)
    r_pix = max(1, int(round(30.0 / pix)))  # 30m / pixel_size
    # construir máscara circular no bloco (lado = 2*r_pix+1)
    size = 2*r_pix + 1
    w2 = Window(col_off=max(col-r_pix,0), row_off=max(row-r_pix,0),
                width=size, height=size)
    block = ds.read(1, window=w2, boundless=True, masked=True).astype("float32")

    # máscara circular (em pixels)
    yy, xx = np.ogrid[:block.shape[0], :block.shape[1]]
    cy, cx = min(r_pix, block.shape[0]-1), min(r_pix, block.shape[1]-1)
    circle = (yy - cy)**2 + (xx - cx)**2 <= (r_pix**2)

    block_np = block.filled(np.nan)
    vals = block_np[circle]
    medb, iqrb = med_iqr(vals)

    return {"2DAP5x5": {"MED": med5, "IQR": iqr5},
            "BUF30m":  {"MED": medb, "IQR": iqrb}}

def main():
    assert os.path.exists(IN_CSV), f"IN_CSV não encontrado: {IN_CSV}"
    assert os.path.exists(MANIFEST), f"Manifesto não encontrado: {MANIFEST}"

    # chaves + coords + manifesto
    hls = pd.read_csv(IN_CSV)
    man = pd.read_csv(MANIFEST)

    keys = ["Sample","Sub_Sample","Date_std"]
    need_cols = keys + ["Lat","Lon"]
    assert all(c in hls.columns for c in need_cols), "Lat/Lon/keys faltando no HLS CSV"
    assert set(keys).issubset(man.columns), "Manifesto sem chaves"
    assert "granule_id" in man.columns, "Manifesto sem granule_id"

    df = hls.merge(man[keys + ["granule_id","download_url"]], on=keys, how="left")

    # selecionar linhas alvo (resume: somente onde ainda há NaN nas colunas de bandas)
    patt = re.compile(r"^(B(\d+|8A))__(2DAP5x5|BUF30m)_(MED|IQR)$")
    band_cols = [c for c in df.columns if patt.match(c)]
    # Se só min set, filtramos apenas B02/B04/B08
    if ONLY_MIN_SET:
        band_cols = [c for c in band_cols if re.match(r"^B0?2__|^B0?4__|^B0?8__", c) or c.startswith("B8A__")]

    mask_need = df[band_cols].isna().any(axis=1)
    idxs = list(df.index[mask_need])
    if LIMIT:
        idxs = idxs[:LIMIT]

    if VERBOSE:
        print(f"[plan] rows total={len(df)} need={mask_need.sum()} running={len(idxs)} bands={'MIN' if ONLY_MIN_SET else 'ALL'}")

    linkcache = _load_linkcache()
    acc_updates = []  # (row_index, dict_values)
    processed = 0

    for k, i in enumerate(idxs, 1):
        row = df.loc[i]
        lon, lat = float(row["Lon"]), float(row["Lat"])

        out_vals = {}
        # processar cada banda necessária
        for b in BANDS:
            sname = short_name(b)  # e.g., B02->B2
            try:
                url = band_url_for_row(row, b, linkcache)
                tif = ensure_download(url)
                with rasterio.open(tif) as ds:
                    stats = stats_on_geoms(ds, lon, lat)
                # preencher para ambas geometrias
                for geom in ("2DAP5x5","BUF30m"):
                    for stat in ("MED","IQR"):
                        col = f"{sname}__{geom}_{stat}"
                        if col in df.columns:
                            out_vals[col] = stats[geom][stat]
            except Exception as e:
                if VERBOSE:
                    print(f"[warn] Sample={row['Sample']}/{row['Sub_Sample']} {row['Date_std']} band={b}: {e}")

        acc_updates.append((i, out_vals))
        processed += 1

        # flush periódico
        if processed % FLUSH_EVERY == 0:
            if acc_updates:
                for idx, vals in acc_updates:
                    for c, v in vals.items():
                        df.at[idx, c] = v
                # write-out seguro
                tmp = OUT_CSV + ".part"
                df.to_csv(tmp, index=False)
                os.replace(tmp, OUT_CSV)
                if VERBOSE:
                    print(f"[flush] wrote batch up to {processed}/{len(idxs)}")
                acc_updates.clear()

    # flush final
    if acc_updates:
        for idx, vals in acc_updates:
            for c, v in vals.items():
                df.at[idx, c] = v
        tmp = OUT_CSV + ".part"
        df.to_csv(tmp, index=False)
        os.replace(tmp, OUT_CSV)
        if VERBOSE:
            print(f"[flush] wrote final batch ({processed}/{len(idxs)})")

    # done
    filled = df[band_cols].notna().sum().sum()
    total  = df[band_cols].size
    print(f"[DONE] filled {filled}/{total} ({filled*100/total:.1f}%) rows_processed={processed}/{len(idxs)} bands={'MIN' if ONLY_MIN_SET else 'ALL'}")

if __name__ == "__main__":
    main()
