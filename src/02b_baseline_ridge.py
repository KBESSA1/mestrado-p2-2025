# -*- coding: utf-8 -*-
import os, argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from utils_lodo import make_lodo_splits, compute_metrics
from feat_picker import pick_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_csv = args.out
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.target_col not in df.columns: raise SystemExit(f"target '{args.target_col}' não existe")
    if args.date_col  not in df.columns:  raise SystemExit(f"date '{args.date_col}' não existe")

    feats = pick_features(df, args.target_col)
    if not feats: raise SystemExit("nenhuma feature espectral/clima encontrada")
    df = df.dropna(subset=feats + [args.target_col]).reset_index(drop=True)

    rows = []
    for fold in make_lodo_splits(df, args.date_col):
        dtag = fold.heldout_date.date().isoformat()
        Xtr = df.loc[fold.train_idx, feats].values
        ytr = df.loc[fold.train_idx, args.target_col].values
        Xte = df.loc[fold.test_idx,  feats].values
        yte = df.loc[fold.test_idx,  args.target_col].values

        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=args.alpha, random_state=0))])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        m = compute_metrics(yte, yhat)
        rows.append({"heldout_date": dtag, "model": "ridge", "r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"]})

    out_df = pd.DataFrame(rows)
    mean_row = {
        "heldout_date": "__mean__", "model": "ridge",
        "r2": float(out_df["r2"].mean()), "rmse": float(out_df["rmse"].mean()), "mae": float(out_df["mae"].mean())
    }
    out_df = pd.concat([out_df, pd.DataFrame([mean_row])], ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"ok -> {out_csv} | feats={len(feats)} | alvo={args.target_col} | alpha={args.alpha}")

if __name__ == "__main__":
    main()
