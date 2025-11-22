# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

"""
Script: 02_baseline_linear.py
Autor: Rodrigo Kbessa (UFMS, 2025)

Descrição rápida:
  - Regressão Linear simples para CP e TDN em LODO (Leave-One-Date-Out).

Por que existe:
  - Serve como régua mínima: se o modelo não ganha disso, não merece
    entrar na conversa da dissertação.

O que faz:
  - Lê um CSV com as features já montadas (RAW/D5/D7, com ou sem clima).
  - Remove chaves e alvos do X, aplica z-score só com base no treino.
  - Roda LODO por Date (cada campanha vira fold de teste).
  - Salva métricas e predições em /workspace/reports para comparação.
"""

# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from utils_lodo import make_lodo_splits, compute_metrics
from feat_picker import pick_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_csv  = args.out
    out_dir  = os.path.dirname(out_csv) or "."
    plots_dir = os.path.join(out_dir, "plots")   # reservado (compat)
    preds_dir = os.path.join(out_dir, "preds")   # reservado (compat)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # ---- carregar e validar
    df = pd.read_csv(args.csv)
    if args.target_col not in df.columns:
        raise SystemExit(f"target '{args.target_col}' não existe")
    if args.date_col not in df.columns:
        raise SystemExit(f"date '{args.date_col}' não existe")

    # ---- features
    feats = pick_features(df, args.target_col)
    if not feats:
        raise SystemExit("nenhuma feature espectral/clima encontrada")
    # limpar NaNs nas colunas usadas
    df = df.dropna(subset=feats + [args.target_col]).reset_index(drop=True)

    # ---- modelo
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression())
    ])

    rows = []
    # LODO: Leave-One-Date-Out
    for fold in make_lodo_splits(df, args.date_col):
        dtag = fold.heldout_date.date().isoformat()

        Xtr = df.loc[fold.train_idx, feats].values
        ytr = df.loc[fold.train_idx, args.target_col].values
        Xte = df.loc[fold.test_idx,  feats].values
        yte = df.loc[fold.test_idx,  args.target_col].values

        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)

        m = compute_metrics(yte, yhat)
        rows.append({
            "heldout_date": dtag,
            "model": "linear",
            "r2": float(m["r2"]),
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"])
        })

    # ---- salvar CSV com média
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        mean_row = {
            "heldout_date": "__mean__",
            "model": "linear",
            "r2": float(out_df["r2"].mean()),
            "rmse": float(out_df["rmse"].mean()),
            "mae": float(out_df["mae"].mean()),
        }
        out_df = pd.concat([out_df, pd.DataFrame([mean_row])], ignore_index=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"ok -> {out_csv} | feats={len(feats)} | plots={plots_dir} | preds={preds_dir} | alvo={args.target_col}")

if __name__ == "__main__":
    main()
