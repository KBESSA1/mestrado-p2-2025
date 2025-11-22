# Diagnóstico — Ganho do clima
Ainda **não** há linhas 'com clima' em metrics_compare.csv.

Quando você rodar os modelos **com clima** e adicionar ao metrics_compare.csv
(com `dataset` no formato `RAW (com clima)`, `D5 (com clima)`, etc.),
o relatório de ganho poderá ser calculado.

## Pares existentes (todos 'sem clima')
| base   | target   | model   |   count | tem_com_clima?   | tem_sem_clima?   |
|:-------|:---------|:--------|--------:|:-----------------|:-----------------|
| D5     | CP       | ridge   |       1 | False            | True             |
| D5     | TDN      | ridge   |       1 | False            | True             |
| RAW    | CP       | gb      |       1 | False            | True             |
| RAW    | CP       | linear  |       1 | False            | True             |
| RAW    | CP       | naive   |       1 | False            | True             |
| RAW    | CP       | xgb     |       1 | False            | True             |
| RAW    | TDN      | gb      |       1 | False            | True             |
| RAW    | TDN      | linear  |       1 | False            | True             |
| RAW    | TDN      | naive   |       1 | False            | True             |
| RAW    | TDN      | xgb     |       1 | False            | True             |
