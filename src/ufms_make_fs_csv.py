# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in", required=True)
    ap.add_argument("--date-col", default="Date")
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--rank-csv", required=True)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--csv-out", required=True)
    ap.add_argument("--features-list-dir", default="/workspace/data/feature_sets")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    ranks = pd.read_csv(args.rank_csv)
    keep_feats = (ranks.sort_values(["mean_gain","freq_top20"], ascending=[False,False])
                        ["feature"].head(args.topk).tolist())

    cols = [c for c in [args.date_col, args.target_col] + keep_feats if c in df.columns]
    out = df[cols].copy()
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv_out, index=False)

    Path(args.features_list_dir).mkdir(parents=True, exist_ok=True)
    list_name = Path(args.csv_out).name.replace(".csv",".features.txt")
    (Path(args.features_list_dir)/list_name).write_text("\n".join(keep_feats), encoding="utf-8")

if __name__ == "__main__":
    main()
