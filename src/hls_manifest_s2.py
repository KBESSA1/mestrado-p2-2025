# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import earthaccess

DF_MAIN = "/workspace/data/data_processed/complete_S2_allgeom.csv"
OUT_DIR = "/workspace/reports/progress"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_MANIFEST = f"{OUT_DIR}/HLS_S30_manifest.csv"

DATE_FMT = "%Y-%m-%d"
DAY_TOLERANCE = int(os.environ.get("HLS_DAY_TOL", "7"))       # janela ± dias
BBOX_PAD_DEG = float(os.environ.get("HLS_BBOX_PAD", "0.02"))  # ~2.2 km

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, DATE_FMT)

def _as_dt(obj):
    if isinstance(obj, datetime):
        return obj
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj.replace("Z",""))
        except Exception:
            return None
    return None

def _granule_dt(g):
    # tenta atributos comuns
    for k in ("datetime","time_start","start_time","BeginningDateTime"):
        v = getattr(g, k, None) if hasattr(g, k) else (getattr(g, "__dict__", {}).get(k))
        if v:
            dt = _as_dt(v)
            if dt:
                return dt
    # tenta dicionários meta/umm
    for dct in (getattr(g,"meta",None), getattr(g,"umm",None)):
        if isinstance(dct, dict):
            for k in ("time_start","start_time","BeginningDateTime"):
                if k in dct:
                    dt = _as_dt(dct[k])
                    if dt:
                        return dt
    return None

def _granule_id(g):
    for k in ("id","granule_id","producer_granule_id","concept_id","title"):
        v = getattr(g, k, None) if hasattr(g, k) else None
        if v: return v
        if isinstance(getattr(g,"meta",None), dict) and k in g.meta: return g.meta[k]
        if isinstance(getattr(g,"umm",None), dict)  and k in g.umm:  return g.umm[k]
    return ""

def _granule_url(g):
    for attr in ("href","download_url","s3_url","data_link"):
        v = getattr(g, attr, None) if hasattr(g, attr) else None
        if v: return v
    for container in (getattr(g,"links",None), getattr(g,"assets",None)):
        if isinstance(container, (list,tuple)):
            for x in container:
                if isinstance(x, dict) and "href" in x:
                    return x["href"]
        if isinstance(container, dict):
            for x in container.values():
                if isinstance(x, dict) and "href" in x:
                    return x["href"]
    return ""

def _granule_cloud(g):
    for src in (getattr(g,"cloud_cover",None),
                getattr(g,"eo_cloud_cover",None),
                getattr(g,"meta",None),
                getattr(g,"umm",None)):
        if isinstance(src, (int,float)): return src
        if isinstance(src, dict):
            for k in ("cloud_cover","eo:cloud_cover","CloudCover"):
                if k in src:
                    try: return float(src[k])
                    except Exception: return None
    return None

def _score(g, target_dt: datetime):
    dt = _granule_dt(g)
    dd = abs((dt - target_dt).days) if isinstance(dt, datetime) else 9999
    cc = _granule_cloud(g)
    cc = cc if isinstance(cc, (int,float)) else 1000
    return (dd, cc)

def _pick_best(cands, target_dt: datetime):
    return sorted(cands, key=lambda x: _score(x, target_dt))[0] if cands else None

def main():
    earthaccess.login(strategy="netrc")
    df = pd.read_csv(DF_MAIN, usecols=["Sample","Sub_Sample","Date_std","Lat","Lon"])

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="search"):
        lat = float(r["Lat"]); lon = float(r["Lon"])
        dt  = parse_date(str(r["Date_std"]))
        t0, t1 = dt - timedelta(days=DAY_TOLERANCE), dt + timedelta(days=DAY_TOLERANCE)
        bbox = (lon - BBOX_PAD_DEG, lat - BBOX_PAD_DEG, lon + BBOX_PAD_DEG, lat + BBOX_PAD_DEG)

        # HLSS30 = HLS Sentinel-2 (harmonizado)
        results = earthaccess.search_data(
            short_name="HLSS30",
            temporal=(t0, t1),
            bounding_box=bbox,
        )

        best = _pick_best(results, dt) if results else None
        if best is None:
            rows.append({
                "Sample": r["Sample"], "Sub_Sample": r["Sub_Sample"], "Date_std": r["Date_std"],
                "Lat": lat, "Lon": lon, "granule_id": "", "granule_time": "",
                "cloud_cover": "", "has_result": False, "download_url": ""
            })
        else:
            gid = _granule_id(best) or ""
            gdt = _granule_dt(best)
            gurl = _granule_url(best) or ""
            gcl = _granule_cloud(best)
            rows.append({
                "Sample": r["Sample"], "Sub_Sample": r["Sub_Sample"], "Date_std": r["Date_std"],
                "Lat": lat, "Lon": lon,
                "granule_id": gid,
                "granule_time": gdt.isoformat() if isinstance(gdt, datetime) else "",
                "cloud_cover": gcl if gcl is not None else "",
                "has_result": True,
                "download_url": gurl
            })

    man = pd.DataFrame(rows)
    man.to_csv(OUT_MANIFEST, index=False)
    ok = int(man["has_result"].sum()) if "has_result" in man else 0
    print(f"[DONE] Manifesto em {OUT_MANIFEST} | cobertura: {ok*100.0/len(man):.1f}% ({ok}/{len(man)})")

if __name__ == "__main__":
    main()
