# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Runner auxiliar para grid search e cross-validation de XGBoost. Usado para explorar melhor o espaço de hiperparâmetros em UFMS.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import argparse, pandas as pd, numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

def _load_xy(csv, date_col, target_col):
    df = pd.read_csv(csv)
    if date_col not in df.columns:
        raise SystemExit(f"coluna de data '{date_col}' não encontrada em {csv}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        raise SystemExit(f"alvo '{target_col}' não é numérico ou não existe.")
    y = num[target_col].astype(float).values
    X = num.drop(columns=[target_col]).fillna(num.median(numeric_only=True)).values
    groups = df[date_col].dt.normalize().values
    return X, y, groups, X.shape[1], len(y)

def _splits(cv_mode, groups, n_splits):
    n = len(groups)
    X_dummy = np.zeros((n,1))
    if cv_mode == "lodo":
        yield from LeaveOneGroupOut().split(X_dummy, None, groups)
    elif cv_mode == "gkfold":
        yield from GroupKFold(n_splits=n_splits).split(X_dummy, None, groups)
    else:
        raise SystemExit("--cv deve ser 'lodo' ou 'gkfold'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", choices=["lodo","gkfold"], default="lodo")
    ap.add_argument("--n_splits", type=int, default=5)
    # XGB params
    ap.add_argument("--num_boost_round", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.06)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--colsample_bytree", type=float, default=0.7)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    args = ap.parse_args()

    X, y, groups, n_feats, n_samples = _load_xy(args.csv, args.date_col, args.target_col)

    params = dict(
        max_depth=args.max_depth,
        eta=args.eta,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        objective="reg:squarederror",
        tree_method="hist",
        # fixa random_state de forma estável
        seed=42
    )

    rows=[]; fold=0
    for tr, te in _splits(args.cv, groups, args.n_splits):
        fold += 1
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dte = xgb.DMatrix(X[te], label=y[te])
        evals = [(dte, "valid")]
        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=args.num_boost_round,
            evals=evals,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=False
        )
        # xgboost 2.x: usar best_iteration (inteiro) se houver ES; senão usar num_boost_round
        best_iter = getattr(booster, "best_iteration", None)
        if best_iter is None:
            # sem early stop: usa todas as árvores
            pred = booster.predict(dte)
        else:
            # prever até best_iteration (inclusivo) → iteration_range usa [ini, fim) ⇒ +1
            pred = booster.predict(dte, iteration_range=(0, best_iter + 1))

        r2  = float(r2_score(y[te], pred))
        rmse= float(np.sqrt(mean_squared_error(y[te], pred)))
        mae = float(mean_absolute_error(y[te], pred))
        rows.append(dict(fold=fold, n_train=len(tr), n_test=len(te), r2=r2, rmse=rmse, mae=mae))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ok -> {out} | feats={n_feats} | samples={n_samples} | cv={args.cv}")

if __name__ == "__main__":
    main()
