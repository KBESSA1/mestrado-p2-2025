import argparse
import os
import pandas as pd
import numpy as np

from utils_lodo import make_lodo_splits, compute_metrics

def build_argparser():
    p = argparse.ArgumentParser(prog="exp01", description="Exp01 S2-paper Δ≤5 (skeleton + naive + inspect)")
    p.add_argument("--csv", type=str, default=None, help="Caminho do CSV real (opcional)")
    p.add_argument("--date-col", type=str, default="date", help="Nome da coluna de data")
    p.add_argument("--target-col", type=str, default="y", help="Nome da coluna alvo")
    p.add_argument("--out", type=str, default="reports/exp01_metrics_demo.csv", help="Arquivo CSV de saída")
    p.add_argument("--demo", action="store_true", help="Rodar com dados sintéticos de demonstração")
    p.add_argument("--inspect", action="store_true", help="Inspecionar CSV (colunas e head) e sair")
    return p

def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        return pd.DataFrame({
            args.date_col: ["2024-01-01","2024-01-01","2024-01-02","2024-01-02","2024-01-03","2024-01-03"],
            "x": [1,2,3,4,5,6],
            args.target_col: [10.0,11.0,12.5,13.0,14.0,15.5],
        })
    if not args.csv:
        raise SystemExit("Informe --csv <arquivo.csv> ou use --demo.")
    return pd.read_csv(args.csv)

def naive_last_value_per_fold(df: pd.DataFrame, date_col: str, y_col: str):
    folds = make_lodo_splits(df, date_col)
    rows = []
    for f in folds:
        train = df.iloc[f.train_idx]
        test  = df.iloc[f.test_idx]
        last_d = pd.to_datetime(train[date_col]).max()
        last_val = train.loc[pd.to_datetime(train[date_col]) == last_d, y_col].mean()
        y_true = test[y_col].to_numpy()
        y_pred = np.full_like(y_true, fill_value=float(last_val), dtype=float)
        m = compute_metrics(y_true, y_pred)
        rows.append({
            "heldout_date": pd.to_datetime(f.heldout_date).date().isoformat(),
            "model": "naive-last",
            **m
        })
    return pd.DataFrame(rows)

def main():
    args = build_argparser().parse_args()
    df = load_data(args)

    if args.inspect:
        print("cols:", list(df.columns))
        print("shape:", df.shape)
        print(df.head(5).to_string(index=False))
        return

    metrics_df = naive_last_value_per_fold(df, args.date_col, args.target_col)
    agg = metrics_df[["r2","rmse","mae"]].mean().to_dict()
    metrics_df.loc[len(metrics_df)] = {"heldout_date":"__mean__", "model":"naive-last", **agg}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    metrics_df.to_csv(args.out, index=False)
    print("ok ->", args.out)

if __name__ == "__main__":
    main()
