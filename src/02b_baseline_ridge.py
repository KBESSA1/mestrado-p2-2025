import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from utils_lodo import make_lodo_splits, compute_metrics

def pick_features(df: pd.DataFrame):
    bands = [f"B{i}" for i in range(1,13)] + ["B8A","B9","B11","B12"]
    idxs  = ["NDVI","NDWI","EVI","LAI","DVI","GCI","GEMI","SAVI"]
    keep  = [c for c in df.columns if c in bands or c in idxs]
    return [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]

def scatter(y, yhat, title, out_png):
    y, yhat = np.asarray(y).ravel(), np.asarray(yhat).ravel()
    lo, hi = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
    pad = 0.05*(hi-lo if hi>lo else 1.0); lo-=pad; hi+=pad
    plt.figure(figsize=(5,5)); plt.scatter(y,yhat,alpha=.7,s=18); plt.plot([lo,hi],[lo,hi],'--',lw=1)
    plt.xlabel("y (obs)"); plt.ylabel("ŷ (pred)"); plt.title(title); plt.xlim(lo,hi); plt.ylim(lo,hi)
    plt.tight_layout(); plt.savefig(out_png,dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", required=True)   # CSV de métricas agregadas
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feats = pick_features(df)
    folds = make_lodo_splits(df, args.date_col)

    out_csv   = args.out
    out_dir   = os.path.dirname(out_csv) or "."
    plots_dir = os.path.join(out_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    preds_dir = os.path.join(out_dir, "preds"); os.makedirs(preds_dir, exist_ok=True)

    results = []
    all_y, all_yhat = [], []
    for k, fold in enumerate(folds, 1):
        tr, te = fold.train_idx, fold.test_idx
        Xtr, Xte = df.iloc[tr][feats].values, df.iloc[te][feats].values
        ytr, yte = df.iloc[tr][args.target_col].values, df.iloc[te][args.target_col].values

        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=args.alpha))])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)

        m = compute_metrics(yte, yhat)
        results.append((fold.heldout_date.date().isoformat(), "ridge", m["r2"], m["rmse"], m["mae"]))

        # salvar preds do fold
        out_pred = os.path.join(preds_dir, f"preds_ridge_{args.target_col}_{fold.heldout_date.date().isoformat()}_a{args.alpha}.csv")
        pd.DataFrame({"y": yte, "yhat": yhat}).to_csv(out_pred, index=False)

        # scatter
        out_png = os.path.join(plots_dir, f"scatter_ridge_{args.target_col}_{fold.heldout_date.date().isoformat()}_a{args.alpha}.png")
        scatter(yte, yhat, f"Ridge a={args.alpha} – {args.target_col} – {fold.heldout_date.date()}", out_png)

        all_y.append(yte); all_yhat.append(yhat)

    # scatter de todos os folds juntos
    all_y = np.concatenate(all_y); all_yhat = np.concatenate(all_yhat)
    scatter(all_y, all_yhat, f"Ridge a={args.alpha} – {args.target_col} – ALL", 
            os.path.join(plots_dir, f"scatter_ridge_{args.target_col}_ALL_a{args.alpha}.png"))

    # salvar métricas + média
    dfm = pd.DataFrame(results, columns=["date","model","r2","rmse","mae"])
    r2m, rmsem, maem = dfm["r2"].mean(), dfm["rmse"].mean(), dfm["mae"].mean()
    dfm = pd.concat([dfm, pd.DataFrame([{"date":"__mean__","model":"ridge","r2":r2m,"rmse":rmsem,"mae":maem}])], ignore_index=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    dfm.to_csv(out_csv, index=False)
    print(f"ok -> {out_csv} | feats={len(feats)} | plots={plots_dir} | preds={preds_dir} | alvo={args.target_col} | alpha={args.alpha}")
    print(f"__mean__,ridge,{r2m},{rmsem},{maem}")

if __name__ == "__main__":
    main()
