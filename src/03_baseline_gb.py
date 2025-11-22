# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Baseline genérico de Gradient Boosting com validação LODO por data. Serve como referência canônica de árvore em todos os cenários.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

"""
Script: 03_baseline_gb.py
Autor: Rodrigo Kbessa (UFMS, 2025)

Descrição rápida:
  - Roda Gradient Boosting para CP e TDN usando validação LODO por data.
  - É um dos scripts oficiais do baseline da dissertação.

O que faz:
  - Lê um CSV preparado (RAW / D5 / D7, com ou sem clima).
  - Separa X e y, monta os folds LODO pela coluna Date.
  - Treina um GB por fold, sem vazamento de campanha.
  - Agrega predições OOF e métricas em /workspace/reports.

Nota do Kbessa:
  - Aqui ficou claro que árvore é "pé no chão" quando o mundo é temporal
    e o número de amostras por data é pequeno.
"""

import sys
import argparse, pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

def pick_features(df, target, date_col):
    drop = {target, date_col, "Satellite_Images_Dates", "Latitude", "Longitude", "SampleID"}
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def lodo_eval(df, date_col, feats, target):
    dates = pd.to_datetime(df[date_col]).dt.date
    X = df[feats].values
    y = df[target].values
    gkf = GroupKFold(n_splits=len(np.unique(dates)))
    rows = []
    for (train_idx, test_idx), d in zip(gkf.split(X, y, groups=dates), np.unique(dates)):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        # dropna por fold
        mtr = ~np.isnan(Xtr).any(axis=1) & ~np.isnan(ytr)
        mte = ~np.isnan(Xte).any(axis=1) & ~np.isnan(yte)
        Xtr, ytr = Xtr[mtr], ytr[mtr]
        Xte, yte = Xte[mte], yte[mte]
        if len(ytr)==0 or len(yte)==0:
            r2 = rmse = mae = np.nan
        else:
            model = GradientBoostingRegressor(
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    subsample=args.subsample,
    max_features=args.max_features,
    random_state=42,
)
            model.set_params(**rparams_gb)

            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = r2_score(yte, pred)
            rmse = mean_squared_error(yte, pred, squared=True) ** 0.5
            mae = mean_absolute_error(yte, pred)
        rows.append({"heldout_date": str(d), "r2": r2, "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    # hiperparâmetros básicos do GB
    ap.add_argument("--n_estimators", type=int, default=400, help="número de árvores do GradientBoosting")
    ap.add_argument("--max_features", default=None, help="estratégia de max_features (ex: 'sqrt', 'log2')")

    # args de I/O / colunas – mesma convenção dos outros baselines
    ap.add_argument("--csv", required=True, help="caminho do CSV de entrada")
    ap.add_argument("--date-col", required=True, help="nome da coluna de data (LODO)")
    ap.add_argument("--target-col", required=True, help="nome da coluna alvo (CP ou TDN)")
    ap.add_argument("--out", required=True, help="arquivo CSV de saída com métricas por data")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.date_col not in df.columns:
        raise SystemExit(f"coluna de data '{args.date_col}' não existe")
    if args.target_col not in df.columns:
        raise SystemExit(f"target '{args.target_col}' não existe")

    feats = pick_features(df, args.target_col, args.date_col)
    res = lodo_eval(df, args.date_col, feats, args.target_col,
                    n_estimators=args.n_estimators,
                    max_features=args.max_features)

    # agrega métrica média no final
    mean_row = {
        "heldout_date": "__mean__",
        "r2":  res["r2"].mean(),
        "rmse": res["rmse"].mean(),
        "mae":  res["mae"].mean(),
    }
    out = pd.concat([res, pd.DataFrame([mean_row])], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"ok -> {args.out} | feats={len(feats)} | alvo={args.target_col}")


if __name__ == "__main__":
    main()
