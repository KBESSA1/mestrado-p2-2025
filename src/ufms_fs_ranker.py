# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

def _load(csv, date_col, target_col):
    df = pd.read_csv(csv)
    if date_col not in df: raise SystemExit(f"date_col '{date_col}' não existe em {csv}")
    if target_col not in df: raise SystemExit(f"target_col '{target_col}' não existe em {csv}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns: raise SystemExit(f"alvo '{target_col}' não é numérico no CSV")
    y = num[target_col].astype(float).values
    X = num.drop(columns=[target_col])
    feats = X.columns.tolist()
    groups = df[date_col].dt.normalize().values
    return X.values.astype(np.float32), y.astype(np.float32), groups, feats

def _splits(groups, cv_mode, n_splits):
    n = len(groups); X_dummy = np.zeros((n,1), dtype=np.float32)
    if cv_mode == "lodo":
        yield from LeaveOneGroupOut().split(X_dummy, None, groups)
    elif cv_mode == "gkfold":
        yield from GroupKFold(n_splits=n_splits).split(X_dummy, None, groups)
    else:
        raise SystemExit("--cv deve ser lodo ou gkfold")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out-rank", required=True)
    ap.add_argument("--cv", choices=["lodo","gkfold"], default="lodo")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--num_boost_round", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.06)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--colsample_bytree", type=float, default=0.7)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    args = ap.parse_args()

    X, y, groups, feats = _load(args.csv, args.date_col, args.target_col)
    ranks = []
    dparams = dict(
        max_depth=args.max_depth, eta=args.eta, subsample=args.subsample,
        colsample_bytree=args.colsample_bytree, reg_lambda=args.reg_lambda,
        objective="reg:squarederror", tree_method="hist", nthread=-1
    )
    folds = list(_splits(groups, args.cv, args.n_splits))
    for fold_id, (tr, te) in enumerate(folds, 1):
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        booster = xgb.train(dparams, dtrain, num_boost_round=args.num_boost_round)
        gain = booster.get_score(importance_type="gain")
        fmap = {f"f{i}": name for i, name in enumerate(feats)}
        for k,v in gain.items():
            ranks.append({"fold": fold_id, "feature": fmap.get(k,k), "gain": float(v)})

    if not ranks: raise SystemExit("Sem importâncias — verifique CSV e parâmetros.")

    df = pd.DataFrame(ranks)
    df["rank_in_fold"] = df.groupby("fold")["gain"].rank(ascending=False, method="dense")
    agg = (df.groupby("feature")
             .agg(mean_gain=("gain","mean"),
                  median_rank=("rank_in_fold","median"),
                  freq_top20=("rank_in_fold", lambda s: (s<=20).mean()),
                  n_folds=("fold","nunique"))
             .reset_index())
    agg = agg.sort_values(["mean_gain","freq_top20"], ascending=[False, False])
    Path(args.out_rank).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_rank, index=False)

if __name__ == "__main__":
    main()
