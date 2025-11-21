# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

def pick_features(df, target_col, date_col):
    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        raise SystemExit(f"alvo '{target_col}' ausente ou não numérico.")
    X = num.drop(columns=[target_col], errors="ignore")
    y = num[target_col].astype(float)
    ok = y.notna()
    return X.loc[ok], y.loc[ok]

def make_model(kind:str, random_state:int=42, ridge_alpha:float=1.0):
    if kind=="naive":
        return None
    if kind=="linear":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LinearRegression())
        ])
    if kind=="ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("rg", Ridge(alpha=ridge_alpha, fit_intercept=True))
        ])
    if kind=="gb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("gb", GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                subsample=1.0, random_state=random_state))
        ])
    if kind=="xgb":
        if not _HAS_XGB:
            raise SystemExit("xgboost não disponível no ambiente.")
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("xgb", XGBRegressor(
                n_estimators=100, max_depth=3, eta=0.1,
                subsample=1.0, colsample_bytree=1.0,
                reg_lambda=1.0, random_state=random_state, n_jobs=-1))
        ])
    if kind=="mlp":
        from sklearn.neural_network import MLPRegressor
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64,32), activation="relu",
                alpha=1e-4, learning_rate_init=1e-3, max_iter=500,
                random_state=random_state))
        ])
    raise SystemExit(f"modelo desconhecido: {kind}")

def run_gkf(csv, date_col, target_col, model_name, out_fp, n_splits=5, ridge_alpha=1.0, random_state=42):
    df = pd.read_csv(csv)
    if date_col not in df.columns:
        raise SystemExit(f"coluna de data '{date_col}' não existe em {csv}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)

    X, y = pick_features(df, target_col, date_col)
    df_use = df.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    groups = df_use[date_col].astype(str).values
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    if model_name=="naive":
        # piso estável: média do y de treino por fold
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            y_tr, y_te = y.iloc[tr].values, y.iloc[te].values
            const = float(np.nanmean(y_tr))
            yhat = np.full_like(y_te, const, dtype=np.float64)
            r2  = float(r2_score(y_te, yhat))
            rmse= float(np.sqrt(mean_squared_error(y_te, yhat)))
            mae = float(mean_absolute_error(y_te, yhat))
            heldout_dates = sorted(df_use.iloc[te][date_col].dt.date.unique().astype(str).tolist())
            rows.append(dict(
                fold=fold, heldout_dates="|".join(heldout_dates),
                n_train=int(len(tr)), n_test=int(len(te)),
                n_features=int(X.shape[1]), model="naive",
                r2=r2, rmse=rmse, mae=mae
            ))
    else:
        model = make_model(model_name, random_state=random_state, ridge_alpha=ridge_alpha)
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
            X_tr, X_te = X.iloc[tr].values, X.iloc[te].values
            y_tr, y_te = y.iloc[tr].values, y.iloc[te].values
            model.fit(X_tr, y_tr)
            yhat = np.asarray(model.predict(X_te)).astype(float)
            r2  = float(r2_score(y_te, yhat))
            rmse= float(np.sqrt(mean_squared_error(y_te, yhat)))
            mae = float(mean_absolute_error(y_te, yhat))
            heldout_dates = sorted(df_use.iloc[te][date_col].dt.date.unique().astype(str).tolist())
            rows.append(dict(
                fold=fold, heldout_dates="|".join(heldout_dates),
                n_train=int(len(tr)), n_test=int(len(te)),
                n_features=int(X.shape[1]), model=model_name,
                r2=r2, rmse=rmse, mae=mae
            ))

    out = pd.DataFrame(rows)
    mean_row = dict(
        fold="__mean__", heldout_dates="__all__",
        n_train=int(out["n_train"].sum()), n_test=int(out["n_test"].sum()),
        n_features=int(out["n_features"].iloc[0]), model=model_name,
        r2=float(out["r2"].mean()), rmse=float(out["rmse"].mean()), mae=float(out["mae"].mean())
    )
    out = pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_fp, index=False)
    print(f"ok -> {out_fp} | feats={X.shape[1]} | alvo={target_col}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--model", required=True, choices=["naive","linear","ridge","gb","xgb","mlp"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()
    run_gkf(args.csv, args.date_col, args.target_col, args.model, args.out,
            n_splits=args.n_splits, ridge_alpha=args.ridge_alpha, random_state=args.random_state)
