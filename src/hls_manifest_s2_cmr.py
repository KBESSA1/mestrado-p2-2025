# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import os, time
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DF_MAIN = "/workspace/data/data_processed/complete_S2_allgeom.csv"
OUT = "/workspace/reports/progress/HLS_S30_manifest.csv"
os.makedirs("/workspace/reports/progress", exist_ok=True)

DATE_FMT = "%Y-%m-%d"
DAY_TOL = int(os.environ.get("HLS_DAY_TOL", "7"))          # ± dias
PAD     = float(os.environ.get("HLS_BBOX_PAD", "0.02"))    # ~2.2 km
SHORT   = "HLSS30"
VER     = "2.0"

# ---------- HTTP session com retries ----------
def make_session():
    s = requests.Session()
    retries = Retry(
        total=3, connect=3, read=3, status=3,
        backoff_factor=0.5,
        status_forcelist=(502,503,504),
        allowed_methods=frozenset(["GET"]),
    )
    s.headers.update({"User-Agent":"UFMS-HLS-Extractor/0.1"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def cmr_ping(sess):
    # Pinga via collections (endpoint válido)
    params = {"short_name": SHORT, "version": VER, "page_size": 1}
    url = "https://cmr.earthdata.nasa.gov/search/collections.json?" + urlencode(params)
    r = sess.get(url, timeout=10)
    r.raise_for_status()
    js = r.json()
    return len((js.get("feed") or {}).get("entry") or []) >= 1

def get_collection_id(sess):
    params = {"short_name": SHORT, "version": VER, "page_size": 5}
    url = "https://cmr.earthdata.nasa.gov/search/collections.json?" + urlencode(params)
    r = sess.get(url, timeout=10)
    r.raise_for_status()
    entries = (r.json().get("feed") or {}).get("entry") or []
    if not entries:
        return None
    # pega o primeiro concept-id
    return entries[0].get("id")

def cmr_search(sess, collection_id, lat, lon, t0, t1, pad=PAD):
    bbox = f"{lon-pad},{lat-pad},{lon+pad},{lat+pad}"
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{t0}T00:00:00Z,{t1}T23:59:59Z",
        "bounding_box": bbox,
        "page_size": 200,
    }
    url = "https://cmr.earthdata.nasa.gov/search/granules.json?" + urlencode(params)
    r = sess.get(url, timeout=15)
    r.raise_for_status()
    js = r.json()
    entries = (js.get("feed") or {}).get("entry") or []
    out = []
    for e in entries:
        gid = e.get("id") or e.get("title") or ""
        tstart = e.get("time_start")
        try:
            dt = datetime.fromisoformat(tstart.replace("Z","")) if tstart else None
        except Exception:
            dt = None
        cc = e.get("cloud_cover")
        dl = None
        for L in e.get("links") or []:
            href = L.get("href")
            if href and href.startswith("http") and "opendap" not in href and "browse" not in href:
                dl = href; break
        out.append({"granule_id": gid, "dt": dt, "cloud_cover": cc, "download_url": dl})
    return out

def pick_best(cands, target_dt):
    def score(c):
        dd = abs((c.get("dt") - target_dt).days) if c.get("dt") else 9999
        cc = c.get("cloud_cover", 1000)
        return (dd, cc)
    return sorted(cands, key=score)[0] if cands else None

def main():
    sess = make_session()
    if not cmr_ping(sess):
        raise SystemExit("[ERRO] CMR ping falhou (coleções HLSS30 v2.0 não retornaram).")

    coll_id = get_collection_id(sess)
    if not coll_id:
        raise SystemExit("[ERRO] Não consegui obter collection_concept_id para HLSS30 v2.0.")

    df = pd.read_csv(DF_MAIN, usecols=["Sample","Sub_Sample","Date_std","Lat","Lon"])
    rows = []
    n = len(df)
    for i, r in df.iterrows():
        lat, lon = float(r["Lat"]), float(r["Lon"])
        dt  = datetime.strptime(str(r["Date_std"]), DATE_FMT)
        t0, t1 = (dt - timedelta(days=DAY_TOL)).strftime("%Y-%m-%d"), (dt + timedelta(days=DAY_TOL)).strftime("%Y-%m-%d")

        try:
            cands = cmr_search(sess, coll_id, lat, lon, t0, t1, PAD)
            best = pick_best(cands, dt)
            rows.append({
                "Sample": r["Sample"], "Sub_Sample": r["Sub_Sample"], "Date_std": r["Date_std"],
                "Lat": r["Lat"], "Lon": r["Lon"],
                "granule_id": best["granule_id"] if best else "",
                "granule_time": best["dt"].isoformat() if (best and best["dt"]) else "",
                "cloud_cover": best.get("cloud_cover") if best else "",
                "has_result": bool(best),
                "download_url": best.get("download_url") if best else ""
            })
        except Exception:
            rows.append({
                "Sample": r["Sample"], "Sub_Sample": r["Sub_Sample"], "Date_std": r["Date_std"],
                "Lat": r["Lat"], "Lon": r["Lon"],
                "granule_id":"", "granule_time":"", "cloud_cover":"",
                "has_result": False, "download_url":""
            })

        if (i+1) % 20 == 0 or (i+1)==n:
            got = sum(int(rr["has_result"]) for rr in rows)
            print(f"[{i+1}/{n}] matches até agora: {got}")

        time.sleep(0.1)  # polidez

    man = pd.DataFrame(rows)
    man.to_csv(OUT, index=False)
    cov = man["has_result"].sum()
    print(f"[DONE] Manifesto em {OUT} | cobertura: {cov}/{len(man)} ({100*cov/len(man):.1f}%)")

if __name__ == "__main__":
    main()
