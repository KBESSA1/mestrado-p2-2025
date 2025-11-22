# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

# -*- coding: utf-8 -*-
import argparse, random, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils_lodo import make_lodo_splits, compute_metrics

# --- seleção de features sem depender de feat_picker
SPECTRAL_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
SPECTRAL_IDXS  = ["NDVI","NDWI","EVI","LAI","DVI","GCI","GEMI","SAVI"]
CLIM_RE = re.compile(r"(TEMP|TMAX|TMIN|PRCP|RAIN|ERA5|RH|WIND|CHIRPS|HUM|PRECIP)", re.I)

def select_features(all_cols, with_climate: bool):
    cols = [c for c in SPECTRAL_BANDS + SPECTRAL_IDXS if c in all_cols]
    if with_climate:
        clim = [c for c in all_cols if CLIM_RE.search(str(c))]
        # evitar duplicata mantendo ordem
        seen = set(cols)
        cols += [c for c in clim if c not in seen]
    return cols

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self,x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-climate", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # seeds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    feats = select_features(df.columns, with_climate=args.with_climate)
    if not feats:
        raise RuntimeError("Nenhuma feature selecionada. Verifique nomes das colunas.")

    rows = []
    for fold in make_lodo_splits(df, date_col=args.date_col):
        tr_idx = np.array(fold.train_idx); te_idx = np.array(fold.test_idx)

        # dropna por fold (features + target)
        use_cols = feats + [args.target_col]
        dtr = df.loc[tr_idx, use_cols].dropna()
        dte = df.loc[te_idx, use_cols].dropna()
        if len(dtr) < 10 or len(dte) < 5:
            # fold muito pequeno — pula mas registra
            rows.append({"heldout_date": getattr(fold, "date", "NA"),
                        "model":"mlp", "r2":np.nan, "rmse":np.nan, "mae":np.nan})
            continue

        Xtr = dtr[feats].astype(np.float32).values
        ytr = dtr[args.target_col].astype(np.float32).values.reshape(-1,1)
        Xte = dte[feats].astype(np.float32).values
        yte = dte[args.target_col].astype(np.float32).values.reshape(-1,1)

        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr).astype(np.float32)
        Xte = scaler.transform(Xte).astype(np.float32)

        model = MLP(Xtr.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True)

        Xte_t = torch.from_numpy(Xte).to(device)
        yte_t = torch.from_numpy(yte).to(device)

        best_rmse = float("inf")
        best_state = None
        waited = 0

        for epoch in range(args.epochs):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            # validação no final do epoch
            model.eval()
            with torch.no_grad():
                y_hat_val = model(Xte_t).detach().cpu().numpy()
                yb_val    = yte_t.detach().cpu().numpy()

            m = compute_metrics(yb_val.reshape(-1), y_hat_val.reshape(-1))
            rmse = float(m["rmse"]); r2 = float(m["r2"]); mae = float(m["mae"])

            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= args.patience:
                    break

        # avalia com melhor estado
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            y_hat = model(Xte_t).detach().cpu().numpy().reshape(-1)
        m_final = compute_metrics(yte.reshape(-1), y_hat)
        rows.append({"heldout_date": getattr(fold, "date", "NA"),
                     "model":"mlp", "r2":float(m_final["r2"]),
                     "rmse":float(m_final["rmse"]), "mae":float(m_final["mae"])})

    # linha de média
    valid = [r for r in rows if np.isfinite(r["rmse"])]
    if valid:
        rows.append({"heldout_date":"__mean__", "model":"mlp",
                     "r2":float(np.mean([r["r2"] for r in valid])),
                     "rmse":float(np.mean([r["rmse"] for r in valid])),
                     "mae":float(np.mean([r["mae"] for r in valid]))})

    out_df = pd.DataFrame(rows, columns=["heldout_date","model","r2","rmse","mae"])
    out_df.to_csv(args.out, index=False)
    print(f"ok -> {args.out} | feats={len(feats)} | clima={int(args.with_climate)} | alvo={args.target_col}")

if __name__ == "__main__":
    main()
