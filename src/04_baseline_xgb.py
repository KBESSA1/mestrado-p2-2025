import argparse, os, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from utils_lodo import make_lodo_splits, compute_metrics

def pick_features(df: pd.DataFrame):
    bands = [f"B{i}" for i in range(1,13)] + ["B8A","B9","B11","B12"]
    indices = ["NDVI","NDWI","EVI","LAI","DVI","GCI","GEMI","SAVI"]
    keep = [c for c in df.columns if (c in bands) or (c in indices)]
    keep = [c for c in keep if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return keep

def scatter_plot(y_true, y_pred, title, out_png):
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    pad = 0.05 * (hi - lo if hi > lo else 1.0); lo -= pad; hi += pad
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.7, s=18)
    plt.plot([lo,hi],[lo,hi],'--',linewidth=1)
    plt.xlabel("y (observado)"); plt.ylabel("ŷ (previsto)"); plt.title(title)
    plt.xlim(lo,hi); plt.ylim(lo,hi); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_csv = args.out
    out_dir   = os.path.dirname(out_csv) or "."
    plots_dir = os.path.join(out_dir, "plots")
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(preds_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    feats = pick_features(df)
    if args.target_col not in df.columns: raise SystemExit(f"target '{args.target_col}' não existe")
    if args.date_col  not in df.columns:  raise SystemExit(f"date '{args.date_col}' não existe")
    if not feats: raise SystemExit("nenhuma feature espectral encontrada")
    df = df.dropna(subset=feats + [args.target_col]).reset_index(drop=True)

    rows, preds_all = [], []
    for fold in make_lodo_splits(df, args.date_col):
        dtag = fold.heldout_date.date().isoformat()
        Xtr = df.loc[fold.train_idx, feats].values
        ytr = df.loc[fold.train_idx, args.target_col].values
        Xte = df.loc[fold.test_idx,  feats].values
        yte = df.loc[fold.test_idx,  args.target_col].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(
                n_estimators=600, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.0, reg_lambda=1.0,
                random_state=42, n_jobs=0, tree_method="hist",
            ))
        ])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)

        m = compute_metrics(yte, yhat)
        rows.append({"fold_date": dtag, "model": "xgb", **m})

        pd.DataFrame({"fold_date": dtag, "y_true": yte, "y_pred": yhat}) \
          .to_csv(os.path.join(preds_dir, f"preds_xgb_{args.target_col}_{dtag}.csv"), index=False)
        scatter_plot(yte, yhat, f"XGB | {args.target_col} | {dtag}",
                     os.path.join(plots_dir, f"scatter_xgb_{args.target_col}_{dtag}.png"))
        preds_all.append(pd.DataFrame({"y_true": yte, "y_pred": yhat}))

    r2m   = float(np.mean([r["r2"]   for r in rows]))
    rmsem = float(np.mean([r["rmse"] for r in rows]))
    maem  = float(np.mean([r["mae"]  for r in rows]))
    rows.append({"fold_date": "__mean__", "model": "xgb", "r2": r2m, "rmse": rmsem, "mae": maem})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if preds_all:
        big = pd.concat(preds_all, ignore_index=True)
        big.to_csv(os.path.join(preds_dir, f"preds_xgb_{args.target_col}_ALL.csv"), index=False)
        scatter_plot(big["y_true"], big["y_pred"], f"XGB | {args.target_col} | TODOS OS FOLDS",
                     os.path.join(plots_dir, f"scatter_xgb_{args.target_col}_ALL.png"))

    print(f"ok -> {out_csv} | feats={len(feats)} | plots={plots_dir} | preds={preds_dir} | alvo={args.target_col}")

if __name__ == "__main__":
    main()
