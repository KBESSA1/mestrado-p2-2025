import argparse
import sys
import pandas as pd
import numpy as np

from utils_lodo import make_lodo_splits  # import relativo ao /workspace/src

def build_argparser():
    p = argparse.ArgumentParser(prog="exp01", description="Exp01 S2-paper Δ≤5 (skeleton)")
    p.add_argument("--csv", type=str, default=None, help="Caminho do CSV real (opcional)")
    p.add_argument("--date-col", type=str, default="date", help="Nome da coluna de data")
    p.add_argument("--target-col", type=str, default="y", help="Nome da coluna alvo")
    p.add_argument("--demo", action="store_true", help="Rodar com dados sintéticos de demonstração")
    return p

def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        # DF sintético só pra validar o pipeline
        df = pd.DataFrame({
            args.date_col: ["2024-01-01","2024-01-01","2024-01-02","2024-01-02","2024-01-03","2024-01-03"],
            "x": [1,2,3,4,5,6],
            args.target_col: [10.0,11.0,12.5,13.0,14.0,15.5],
        })
        return df
    if not args.csv:
        raise SystemExit("Informe --csv <arquivo.csv> ou use --demo.")
    return pd.read_csv(args.csv)

def main():
    args = build_argparser().parse_args()
    print("exp01: args ok")
    df = load_data(args)
    print(f"dataset shape: {df.shape}")
    if args.date_col not in df.columns:
        raise SystemExit(f"Coluna de data '{args.date_col}' não encontrada nas colunas: {list(df.columns)}")
    folds = make_lodo_splits(df, args.date_col)
    print(f"folds LODO: {len(folds)} | held-out (primeiras 3): {[f.heldout_date.date().isoformat() for f in folds[:3]]}")
    print("skeleton finalizado ✔")

if __name__ == "__main__":
    main()


