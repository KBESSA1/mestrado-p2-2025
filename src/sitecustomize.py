# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Script de suporte dentro do projeto UFMS-pastagens. Responsável por alguma etapa de pré-processamento, experimento ou análise.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.


# sitecustomize: carregado automaticamente; faz monkeypatch nos modelos
import sys, argparse

# -------- GradientBoostingRegressor (sklearn) --------
def _gb_read_flags(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--n_estimators", type=int)
    p.add_argument("--max_depth", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--subsample", type=float)
    p.add_argument("--max_features")
    opts, _ = p.parse_known_args(argv)
    return opts

try:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    _GBR_orig_init = _GBR.__init__
    def _GBR_init(self, *a, **kw):
        opts = _gb_read_flags(sys.argv[1:])
        if opts.n_estimators is not None: kw["n_estimators"] = opts.n_estimators
        if opts.max_depth    is not None: kw["max_depth"]    = opts.max_depth
        if opts.learning_rate is not None: kw["learning_rate"] = opts.learning_rate
        if opts.subsample    is not None: kw["subsample"]    = opts.subsample
        if opts.max_features is not None: kw["max_features"] = opts.max_features
        if "random_state" not in kw: kw["random_state"] = 42
        return _GBR_orig_init(self, *a, **kw)
    _GBR.__init__ = _GBR_init
except Exception:
    pass

# -------- XGBRegressor (xgboost) --------
def _xgb_read_flags(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--n_estimators", type=int)
    p.add_argument("--max_depth", type=int)
    p.add_argument("--eta", type=float)
    p.add_argument("--subsample", type=float)
    p.add_argument("--colsample_bytree", type=float)
    p.add_argument("--reg_lambda", type=float)
    opts, _ = p.parse_known_args(argv)
    return opts

try:
    import xgboost as xgb
    _XGB_orig_init = xgb.XGBRegressor.__init__
    def _XGB_init(self, *a, **kw):
        opts = _xgb_read_flags(sys.argv[1:])
        if opts.n_estimators     is not None: kw["n_estimators"]      = opts.n_estimators
        if opts.max_depth        is not None: kw["max_depth"]         = opts.max_depth
        if opts.eta              is not None: kw["learning_rate"]     = opts.eta
        if opts.subsample        is not None: kw["subsample"]         = opts.subsample
        if opts.colsample_bytree is not None: kw["colsample_bytree"]  = opts.colsample_bytree
        if opts.reg_lambda       is not None: kw["reg_lambda"]        = opts.reg_lambda
        return _XGB_orig_init(self, *a, **kw)
    xgb.XGBRegressor.__init__ = _XGB_init
except Exception:
    pass
