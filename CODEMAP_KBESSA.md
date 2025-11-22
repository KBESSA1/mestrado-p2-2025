# Mapa de Código do Projeto (vista Kbessa)

Mapa automático de tudo que existe em `src/` dentro do container.

## Baselines oficiais gerais (Linear, Ridge, GB, XGB, MLP)

- `02_baseline_linear.py`
- `02b_baseline_ridge.py`
- `03_baseline_gb.py`
- `04_baseline_xgb.py`
- `05_baseline_mlp.py`

## Experimentos LODO por cenário (CP/TDN, D5/D7, clim/noclim, modelo)

- `01_cp_D5_clim_gb.py`
- `01_cp_D5_clim_kan.py`
- `01_cp_D5_clim_linear.py`
- `01_cp_D5_clim_mlp.py`
- `01_cp_D5_clim_naive.py`
- `01_cp_D5_clim_ridge.py`
- `01_cp_D5_clim_xgb.py`
- `01_cp_D5_clim_xnet.py`
- `01_cp_D5_noclim_gb.py`
- `01_cp_D5_noclim_kan.py`
- `01_cp_D5_noclim_linear.py`
- `01_cp_D5_noclim_lodo.py`
- `01_cp_D5_noclim_mlp.py`
- `01_cp_D5_noclim_naive.py`
- `01_cp_D5_noclim_ridge.py`
- `01_cp_D5_noclim_xgb.py`
- `01_cp_D5_noclim_xnet.py`
- `01_cp_D7_clim_gb.py`
- `01_cp_D7_clim_kan.py`
- `01_cp_D7_clim_linear.py`
- `01_cp_D7_clim_mlp.py`
- `01_cp_D7_clim_naive.py`
- `01_cp_D7_clim_ridge.py`
- `01_cp_D7_clim_xgb.py`
- `01_cp_D7_clim_xnet.py`
- `01_cp_D7_noclim_gb.py`
- `01_cp_D7_noclim_kan.py`
- `01_cp_D7_noclim_lodo.py`
- `01_cp_D7_noclim_mlp.py`
- `01_cp_D7_noclim_naive.py`
- `01_cp_D7_noclim_ridge.py`
- `01_cp_D7_noclim_xgb.py`
- `01_cp_D7_noclim_xnet.py`
- `01_exp_s2paper_d5_baselines.py`
- `01_tdn_D5_clim_gb.py`
- `01_tdn_D5_clim_kan.py`
- `01_tdn_D5_clim_linear.py`
- `01_tdn_D5_clim_mlp.py`
- `01_tdn_D5_clim_naive.py`
- `01_tdn_D5_clim_ridge.py`
- `01_tdn_D5_clim_xgb.py`
- `01_tdn_D5_clim_xnet.py`
- `01_tdn_D5_noclim_gb.py`
- `01_tdn_D5_noclim_kan.py`
- `01_tdn_D5_noclim_linear.py`
- `01_tdn_D5_noclim_mlp.py`
- `01_tdn_D5_noclim_naive.py`
- `01_tdn_D5_noclim_ridge.py`
- `01_tdn_D5_noclim_xgb.py`
- `01_tdn_D5_noclim_xnet.py`
- `01_tdn_D7_clim_gb.py`
- `01_tdn_D7_clim_kan.py`
- `01_tdn_D7_clim_linear.py`
- `01_tdn_D7_clim_mlp.py`
- `01_tdn_D7_clim_naive.py`
- `01_tdn_D7_clim_ridge.py`
- `01_tdn_D7_clim_xgb.py`
- `01_tdn_D7_clim_xnet.py`
- `01_tdn_D7_noclim_gb.py`
- `01_tdn_D7_noclim_kan.py`
- `01_tdn_D7_noclim_linear.py`
- `01_tdn_D7_noclim_lodo.py`
- `01_tdn_D7_noclim_mlp.py`
- `01_tdn_D7_noclim_naive.py`
- `01_tdn_D7_noclim_ridge.py`
- `01_tdn_D7_noclim_xgb.py`
- `01_tdn_D7_noclim_xnet.py`

## Funções utilitárias / helpers gerais

- `00_utils_lodo.py`
- `__pycache__/utils_lodo.cpython-310.pyc`
- `utils_lodo.py`

## Outros arquivos

- `03_baseline_gb.py.bak`
- `04_baseline_xgb.py.bak`
- `__pycache__/feat_picker.cpython-310.pyc`
- `__pycache__/feature_config.cpython-310.pyc`
- `__pycache__/sitecustomize.cpython-310.pyc`
- `lit_prior.yaml`

## Outros scripts Python de suporte

- `05_baseline_mlp.backup.preproc.py`
- `05_baseline_mlp.backup.py`
- `__init__.py`
- `_compare_lodo_vs_gkf.py`
- `_eval_groupkfold.py`
- `compare_climate_gains.py`
- `feat_picker.py`
- `feature_config.py`
- `lit_feature_filter.py`
- `ping.py`
- `progress_dashboard.py`
- `quick_test.py`
- `run_exp01.py`
- `run_with_climate_raw.py`
- `sitecustomize.py`

## Runners auxiliares de cross-validation / grid search

- `__pycache__/_gb_cv_runner.cpython-310.pyc`
- `__pycache__/_hgb_cv_runner.cpython-310.pyc`
- `__pycache__/_mlp_cv_runner.cpython-310.pyc`
- `__pycache__/_xgb_cv_runner.cpython-310.pyc`
- `__pycache__/_xgb_native_cv_runner.cpython-310.pyc`
- `_gb_cv_runner.py`
- `_hgb_cv_runner.py`
- `_mlp_cv_runner.py`
- `_xgb_cv_runner.py`
- `_xgb_cv_runner_native.py`
- `_xgb_native_cv_runner.py`

## Scripts de HLS / Sentinel / pré-processamento de bandas

- `datasets/__pycache__/make_hls_s2_bands.cpython-310.pyc`
- `datasets/make_hls_s2_bands.backup.py`
- `datasets/make_hls_s2_bands.procpatch.backup.py`
- `datasets/make_hls_s2_bands.py`
- `datasets/make_hls_s2_bands.qcpatch.backup.py`
- `hls_extract_fill_bands.py`
- `hls_extract_fill_bands_v2.py`
- `hls_manifest_s2.py`
- `hls_manifest_s2_cmr.py`
- `make_hls_datasets.py`

## Scripts de seleção/importância de features (FS via XGBoost / Ridge)

- `02_cp_D5_clim_ridge_fs.py`
- `02_cp_D5_noclim_ridge_fs.py`
- `02_cp_D7_clim_ridge_fs.py`
- `02_cp_D7_noclim_ridge_fs.py`
- `02_feature_importance_xgb_all.py`
- `02_tdn_D5_clim_ridge_fs.py`
- `02_tdn_D5_noclim_ridge_fs.py`
- `02_tdn_D7_clim_ridge_fs.py`
- `02_tdn_D7_noclim_ridge_fs.py`
- `03_cp_D5_clim_gb_fs.py`
- `03_cp_D5_noclim_gb_fs.py`
- `03_cp_D7_clim_gb_fs.py`
- `03_cp_D7_noclim_gb_fs.py`
- `03_feature_importance_summary.py`
- `03_tdn_D5_clim_gb_fs.py`
- `03_tdn_D5_noclim_gb_fs.py`
- `03_tdn_D7_clim_gb_fs.py`
- `03_tdn_D7_noclim_gb_fs.py`
- `__pycache__/ufms_fs_ranker.cpython-310.pyc`
- `__pycache__/ufms_make_fs_csv.cpython-310.pyc`
- `ufms_fs_ranker.py`
- `ufms_make_fs_csv.py`

## Shell scripts de automação de experimentos

- `make_policy_symlinks.sh`
- `run_final_baselines.sh`

