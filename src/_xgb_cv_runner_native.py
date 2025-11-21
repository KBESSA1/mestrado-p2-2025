# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Runner auxiliar para grid search e cross-validation de XGBoost. Usado para explorar melhor o espaço de hiperparâmetros em UFMS.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import argparse, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, KFold
import xgboost as xgb

def _load_xy(csv, date_col, target_col):
    df = pd.read_csv(csv)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        raise SystemExit(f"alvo '{target_col}' não é numérico ou não existe.")
    y = num[target_col].astype(float)
    X = num.drop(columns=[target_col])
    ok = y.notna()
    X, y = X.loc[ok], y.loc[ok]
    X = X.fillna(X.median(numeric_only=True))
    return X.values, y.values, df

def _cv_splits(df, date_col, n_splits):
    return TimeSeriesSplit(n_splits=n_splits) if date_col in df.columns else KFold(n_splits=n_splits, shuffle=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    # Hiperparâmetros equivalentes ao seu grid:
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.06)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--colsample_bytree", type=float, default=0.7)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--tree_method", type=str, default="hist")  # "hist" (CPU) | "gpu_hist" (GPU)
    args = ap.parse_args()

    X, y, df = _load_xy(args.csv, args.date_col, args.target_col)
    cv = _cv_splits(df, args.date_col, args.n_splits)

    rows, fold = [], 0
    for tr, te in cv.split(X):
        fold += 1
        dtrain = xgb.DMatrix(X[tr], label=y[tr])
        dtest  = xgb.DMatrix(X[te],  label=y[te])
        params = {
            "objective": "reg:squarederror",
            "eta": args.eta,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "lambda": args.reg_lambda,
            "tree_method": args.tree_method,
            "seed": 42,
        }
        bst = xgb.train(params, dtrain, num_boost_round=args.n_estimators)
        p = bst.predict(dtest)
        r2   = float(r2_score(y[te], p))
        rmse = float(np.sqrt(mean_squared_error(y[te], p)))
        mae  = float(mean_absolute_error(y[te], p))
        rows.append(dict(fold=fold, r2=r2, rmse=rmse, mae=mae))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ok -> {out} | feats={X.shape[1]} | alvo={args.target_col}")

if __name__ == "__main__":
    main()
