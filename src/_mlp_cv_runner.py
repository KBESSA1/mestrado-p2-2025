# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

torch.manual_seed(42)
np.random.seed(42)

def pick_features(df, date_col, target_col):
    # usa TODAS as numéricas exceto data/target/ids comuns
    drop = {target_col}
    num = df.select_dtypes(include=[np.number]).copy()
    feats = [c for c in num.columns if c not in drop]
    return feats

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)

def train_epoch(model, opt, loss_fn, X, y, bs=64):
    model.train(); tot=0.0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float()
        yb = torch.from_numpy(y[i:i+bs]).float().view(-1,1)
        opt.zero_grad(); pred = model(xb); loss = loss_fn(pred, yb)
        loss.backward(); opt.step(); tot += loss.item()*(len(xb))
    return tot/len(X)

@torch.no_grad()
def eval_rmse(model, X, y, bs=64):
    model.eval(); loss_fn = nn.MSELoss(reduction="sum"); sse=0.0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float()
        yb = torch.from_numpy(y[i:i+bs]).float().view(-1,1)
        pred = model(xb); sse += loss_fn(pred, yb).item()
    return np.sqrt(sse/len(X))

def metrics(y_true, y_pred):
    y = y_true; yhat = y_pred
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    rmse = np.sqrt(np.mean((y - yhat)**2))
    mae = np.mean(np.abs(y - yhat))
    return r2, rmse, mae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", choices=["lodo","gkfold"], default="lodo")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.date_col not in df: raise SystemExit(f"date col '{args.date_col}' não existe")
    if args.target_col not in df: raise SystemExit(f"target '{args.target_col}' não existe")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.sort_values(args.date_col).reset_index(drop=True)

    feats = pick_features(df, args.date_col, args.target_col)
    X_all = df[feats].to_numpy(dtype=np.float32)
    y_all = df[args.target_col].to_numpy(dtype=np.float32)
    groups = df[args.date_col].dt.normalize().values

    # splits
    if args.cv == "lodo":
        splitter = LeaveOneGroupOut().split(np.zeros((len(groups),1)), None, groups)
    else:
        splitter = GroupKFold(n_splits=args.n_splits).split(np.zeros((len(groups),1)), None, groups)

    rows = []
    for fold_id, (tr, te) in enumerate(splitter, 1):
        Xtr_raw, ytr = X_all[tr], y_all[tr]
        Xte_raw, yte = X_all[te], y_all[te]

        # imputer + scaler por fold
        pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ])
        Xtr = pipe.fit_transform(Xtr_raw)
        Xte = pipe.transform(Xte_raw)

        model = MLP(Xtr.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        best = np.inf; bad=0
        for ep in range(1, args.epochs+1):
            _ = train_epoch(model, opt, loss_fn, Xtr, ytr, bs=64)
            rmse = eval_rmse(model, Xte, yte, bs=256)
            if rmse < best - 1e-6:
                best = rmse; bad = 0; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= args.patience:
                break
        # restaura best
        model.load_state_dict(best_state)

        with torch.no_grad():
            yhat = model(torch.from_numpy(Xte).float()).cpu().numpy().ravel()
        r2, rmse, mae = metrics(yte, yhat)
        heldout_date = pd.to_datetime(df.loc[te, args.date_col]).dt.normalize().unique()
        heldout_str = heldout_date[0].strftime("%Y-%m-%d") if len(heldout_date) else "NA"
        rows.append({"fold": heldout_str, "r2": r2, "rmse": rmse, "mae": mae,
                     "n_train": int(len(tr)), "n_test": int(len(te)), "feats": int(Xtr.shape[1])})

    out = pd.DataFrame(rows)
    if not out.empty:
        out.loc[len(out)] = {
            "fold":"__mean__", "r2": out["r2"].mean(), "rmse": out["rmse"].mean(), "mae": out["mae"].mean(),
            "n_train": out["n_train"].mean(), "n_test": out["n_test"].mean(), "feats": out["feats"].iloc[0]
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"ok -> {args.out} | feats={rows[0]['feats'] if rows else 'NA'} | folds={len(rows)} | cv={args.cv}")

if __name__ == "__main__":
    main()
