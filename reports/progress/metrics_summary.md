# Métricas LODO (linha __mean__) — Resumo por alvo e modelo

| target   | model_std   |        r2 |    rmse |     mae | source_file                  | model_raw   |
|:---------|:------------|----------:|--------:|--------:|:-----------------------------|:------------|
| CP       | Naïve       | -2.34611  | 2.73819 | 2.33037 | exp01_metrics_CP_naive.csv   | naive-last  |
| CP       | Linear      | -2.70449  | 2.57597 | 2.15877 | exp01_metrics_CP_linear.csv  | linear      |
| CP       | Ridge       | -2.62677  | 2.57577 | 2.24143 | exp01_metrics_CP_ridge.csv   | ridge       |
| CP       | GB          | -1.06073  | 2.3176  | 1.95905 | exp01_metrics_CP_gb.csv      | gb          |
| CP       | XGB         | -1.05056  | 2.31535 | 1.97705 | exp01_metrics_CP_xgb.csv     | xgb         |
| TDN      | Naïve       | -0.450178 | 5.20624 | 4.31886 | exp01_metrics_TDN_naive.csv  | naive-last  |
| TDN      | Linear      | -3.63848  | 7.06856 | 5.62038 | exp01_metrics_TDN_linear.csv | linear      |
| TDN      | Ridge       | -2.10328  | 6.11476 | 5.09897 | exp01_metrics_TDN_ridge.csv  | ridge       |
| TDN      | GB          | -0.734061 | 5.49295 | 4.57143 | exp01_metrics_TDN_gb.csv     | gb          |
| TDN      | XGB         | -0.853355 | 5.68684 | 4.72867 | exp01_metrics_TDN_xgb.csv    | xgb         |
