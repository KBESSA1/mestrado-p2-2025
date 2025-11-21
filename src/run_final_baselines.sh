#!/usr/bin/env bash
set -euo pipefail

OUTDIR="/workspace/reports/exp03"
mkdir -p "$OUTDIR"

for F in /workspace/data/policy_selected/ufms_*_for_*_*.csv; do
  BN=$(basename "$F")  # ufms_{BASE}_{SOURCE}_for_{TARGET}_{MODEL}.csv

  BASE=$(echo "$BN" | cut -d'_' -f2)
  SOURCE=$(echo "$BN" | cut -d'_' -f3)

  # MODEL = sufixo antes de .csv
  MODEL="${BN%.csv}"
  MODEL="${MODEL##*_}"

  # TARGET = trecho entre '_for_' e '_{MODEL}.csv' (pode conter underscores)
  TARGET="${BN#*_for_}"                # remove prefixo até 'for_'
  TARGET="${TARGET%_${MODEL}.csv}"     # remove sufixo '_MODEL.csv'

  OUT="${OUTDIR}/exp03_${BASE}_${TARGET}_${MODEL}_${SOURCE}_lodo_FINAL.csv"
  echo "[RUN] ${BASE}/${SOURCE}/${TARGET}/${MODEL} -> ${OUT}"

  case "$MODEL" in
    hgb)
      python3 /workspace/src/_hgb_cv_runner.py \
        --csv "$F" --date-col Date --target-col "$TARGET" \
        --out "$OUT" --cv lodo
      ;;
    xgbnative)
      python3 /workspace/src/_xgb_native_cv_runner.py \
        --csv "$F" --date-col Date --target-col "$TARGET" \
        --out "$OUT" --cv lodo \
        --num_boost_round 800 --max_depth 4 --eta 0.06 \
        --subsample 0.7 --colsample_bytree 0.7 --reg_lambda 1.0
      ;;
    mlp)
      python3 /workspace/src/_mlp_cv_runner.py \
        --csv "$F" --date-col Date --target-col "$TARGET" \
        --out "$OUT" --cv lodo
      ;;
    *)
      echo "[SKIP] modelo desconhecido: $MODEL"
      ;;
  esac
done
