# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Runner auxiliar para grid search e cross-validation de Gradient Boosting. Pensado para tunar hiperparâmetros sem mexer nos scripts principais.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

"""
Script: _gb_cv_runner.py
Autor: Rodrigo Kbessa (UFMS, 2025)

Descrição rápida:
  - Runner auxiliar para grid leve de Gradient Boosting.

Uso:
  - Não é script "oficial" da dissertação, é ferramenta de bastidor.
  - Ajuda a testar combinações simples de n_estimators, depth, etc.,
    sem mexer nos scripts 03/04 originais.
"""

import argparse, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
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
    groups = df[date_col].dt.normalize()  # agrupa por dia/campanha
    return X.values, y.values, groups.values, X.shape[1], len(y)

def _get_cv(cv_mode, groups, n_splits):
    n = len(groups)
    # X_dummy atende a API do sklearn (precisa de array-like)
    X_dummy = np.zeros((n, 1), dtype=float)
    if cv_mode == "lodo":
        return LeaveOneGroupOut().split(X_dummy, None, groups)
    elif cv_mode == "gkfold":
        gkf = GroupKFold(n_splits=n_splits)
        return gkf.split(X_dummy, None, groups)
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
    # hiperparâmetros
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.06)
    ap.add_argument("--subsample", type=float, default=0.7)
    ap.add_argument("--max_features", default="sqrt")
    args = ap.parse_args()

    X, y, groups, n_feats, n_samples = _load_xy(args.csv, args.date_col, args.target_col)
    rows, fold = [], 0
    for tr_idx, te_idx in _get_cv(args.cv, groups, args.n_splits):
        fold += 1
        m = GradientBoostingRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            max_features=None if args.max_features in [None,"None","none",""] else args.max_features,
            random_state=42,
        )
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict(X[te_idx])
        r2  = float(r2_score(y[te_idx], p))
        rmse= float(np.sqrt(mean_squared_error(y[te_idx], p)))
        mae = float(mean_absolute_error(y[te_idx], p))
        rows.append(dict(fold=fold, n_train=len(tr_idx), n_test=len(te_idx), r2=r2, rmse=rmse, mae=mae))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"ok -> {out} | feats={n_feats} | samples={n_samples} | cv={args.cv}")

if __name__ == "__main__":
    main()
