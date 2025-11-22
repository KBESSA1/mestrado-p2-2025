#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/workspace/src:${PYTHONPATH:-}

BASES=("RAW" "D5" "D7")
SOURCES=("noclim" "clim")
TARGETS=("CP" "TDN_based_ADF")
MODELS=("naive" "linear" "ridge" "gb" "xgb" "mlp")

for B in "${BASES[@]}"; do
  for S in "${SOURCES[@]}"; do
    CSV="/workspace/data/data_processed/ufms_${B}_${S}.csv"
    if [[ ! -f "$CSV" ]]; then
      echo "(skip) não achei $CSV"
      continue
    fi
    for T in "${TARGETS[@]}"; do
      for M in "${MODELS[@]}"; do
        OUT="/workspace/reports/exp01_${B}_${T}_${M}_${S}_gkf.csv"
        /usr/bin/python3 /workspace/src/_eval_groupkfold.py \
          --csv "$CSV" --date-col Date --target-col "$T" \
          --model "$M" --out "$OUT" --n_splits 5 --ridge_alpha 1.0 --random_state 42
      done
    done
  done
done

# comparativo LODO × GKF (gera summary e arquivos por cenário)
mkdir -p /workspace/reports/progress
/usr/bin/python3 /workspace/src/_compare_lodo_vs_gkf.py
