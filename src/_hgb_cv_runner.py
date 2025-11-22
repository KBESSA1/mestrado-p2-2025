# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import argparse, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
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
    y = num[target_col].astype(float)
    X = num.drop(columns=[target_col])
    ok = y.notna()
    X, y, df = X.loc[ok], y.loc[ok], df.loc[ok]
    X = X.fillna(X.median(numeric_only=True))
    groups = df[date_col].dt.normalize()
    return X.values, y.values, groups.values, X.shape[1], len(y)

def _splits(cv_mode, groups, n_splits):
    n = len(groups)
    X_dummy = np.zeros((n,1), dtype=float)
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
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--learning_rate", type=float, default=0.06)
    ap.add_argument("--max_leaf_nodes", type=int, default=31)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--l2_regularization", type=float, default=0.0)
    args = ap.parse_args()

    X, y, groups, n_feats, n_samples = _load_xy(args.csv, args.date_col, args.target_col)
    rows, fold = [], 0
    for tr, te in _splits(args.cv, groups, args.n_splits):
        fold += 1
        m = HistGradientBoostingRegressor(
            max_iter=args.max_iter, learning_rate=args.learning_rate,
            max_leaf_nodes=args.max_leaf_nodes, max_depth=args.max_depth,
            l2_regularization=args.l2_regularization, random_state=42)
        m.fit(X[tr], y[tr]); p = m.predict(X[te])
        r2  = float(r2_score(y[te], p))
        rmse= float(np.sqrt(mean_squared_error(y[te], p)))
        mae = float(mean_absolute_error(y[te], p))
        rows.append(dict(fold=fold, n_train=len(tr), n_test=len(te), r2=r2, rmse=rmse, mae=mae))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ok -> {out} | feats={n_feats} | samples={n_samples} | cv={args.cv}")

if __name__ == "__main__":
    main()
