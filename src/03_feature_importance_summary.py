# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

import pandas as pd
from pathlib import Path

IN_CSV = Path("/workspace/reports/feature_importance/feature_importance_xgb_all_scenarios.csv")
OUT_DIR = Path("/workspace/reports/feature_importance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 15  # top-K por cenário

def main():
    print(f"[INFO] Lendo {IN_CSV}")
    df = pd.read_csv(IN_CSV)

    # Sanidade básica
    expected_cols = {"scenario", "target_col", "delta_max_days", "use_climate", "rank", "feature", "importance"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando no CSV global: {missing}")

    # 1) Top-K por cenário (tabela para você consultar)
    topk = (
        df[df["rank"] <= TOP_K]
        .sort_values(["target_col", "scenario", "rank"])
        .reset_index(drop=True)
    )

    out_topk = OUT_DIR / f"feature_importance_xgb_top{TOP_K}_by_scenario.csv"
    topk.to_csv(out_topk, index=False)
    print(f"[OK] Top-{TOP_K} por cenário salvo em: {out_topk}")

    # 2) Estabilidade por target (CP e TDN), agregando cenários
    #    Conta quantas vezes a feature aparece no top-K e a média de rank/importance.
    results = []

    for target in ["CP", "TDN_based_ADF"]:
        df_t = df[df["target_col"] == target].copy()
        df_t_topk = df_t[df_t["rank"] <= TOP_K].copy()

        if df_t_topk.empty:
            print(f"[AVISO] Nenhuma feature no top-{TOP_K} para target={target}? Verificar dados.")
            continue

        # Quantos cenários distintos existem para esse target
        n_scenarios = df_t["scenario"].nunique()

        # Agrupar por feature
        grp = df_t_topk.groupby("feature")
        agg = grp.agg(
            times_in_topK=("scenario", "nunique"),
            mean_rank=("rank", "mean"),
            mean_importance=("importance", "mean"),
        ).reset_index()

        # Normalizar por nº de cenários (fração de cenários em que aparece no top-K)
        agg["freq_in_topK"] = agg["times_in_topK"] / n_scenarios

        # Ordenar por: (1) freq em topK, (2) mean_rank, (3) mean_importance
        agg = agg.sort_values(
            by=["freq_in_topK", "times_in_topK", "mean_rank", "mean_importance"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

        # Salvar separado por target
        out_target = OUT_DIR / f"feature_importance_xgb_stability_{target}.csv"
        agg.to_csv(out_target, index=False)
        print(f"[OK] Estabilidade de features (target={target}) salva em: {out_target}")

if __name__ == "__main__":
    main()
