#!/usr/bin/env bash
set -euo pipefail

POL="/workspace/reports/progress/UFMS_FS15_policy.csv"
OUTDIR="/workspace/data/policy_selected"
mkdir -p "$OUTDIR"

python3 - <<'PY'
import pandas as pd, os, pathlib, sys
pol = pd.read_csv("/workspace/reports/progress/UFMS_FS15_policy.csv")
outdir = "/workspace/data/policy_selected"
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

def src_path(base, source, target, variant):
    if variant == "full":
        return f"/workspace/data/data_processed/ufms_{base}_{source}.csv"
    elif variant == "fs15":
        return f"/workspace/data/feature_selected/ufms_{base}_{source}_FS15_for_{target}.csv"
    else:
        raise SystemExit(f"variant desconhecido: {variant}")

rows = []
for _,r in pol.iterrows():
    base, source, target, model, variant = r["base"], r["source"], r["target"], r["model"], r["use_variant"]
    src = src_path(base, source, target, variant)
    if not os.path.exists(src):
        print(f"[WARN] missing: {src}", file=sys.stderr)
        continue
    link = f"{outdir}/ufms_{base}_{source}_for_{target}_{model}.csv"
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(src, link)
        print("link ->", os.path.basename(link), "->", src)
    except OSError as e:
        print(f"[ERR] {link}: {e}", file=sys.stderr)
PY
