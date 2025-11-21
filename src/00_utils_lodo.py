# [kbessa-header]
# Autor: Rodrigo Kbessa (UFMS) – projeto pastagens tropicais
# Notas: Utilitário de validação Leave-One-Date-Out (LODO). Simula cenário real de prever campanhas futuras nunca vistas.
# Observação: comentários escritos no espírito de diário de bordo do mestrado.

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    heldout_date: pd.Timestamp

def make_lodo_splits(df: pd.DataFrame, date_col: str) -> List[Fold]:
    if date_col not in df.columns:
        raise KeyError(f"Coluna '{date_col}' não existe no DataFrame.")
    dates = pd.to_datetime(df[date_col]).values
    unique_dates = np.unique(dates)
    folds: List[Fold] = []
    for d in unique_dates:
        test_mask = dates == d
        train_mask = ~test_mask
        folds.append(Fold(
            train_idx=np.nonzero(train_mask)[0],
            test_idx=np.nonzero(test_mask)[0],
            heldout_date=pd.to_datetime(d),
        ))
    folds.sort(key=lambda f: f.heldout_date)
    return folds

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }

if __name__ == "__main__":
    df = pd.DataFrame({
        "date": ["2024-01-01","2024-01-01","2024-01-02","2024-01-02","2024-01-03","2024-01-03"],
        "x": [1,2,3,4,5,6],
        "y": [10.0,11.0,12.5,13.0,14.0,15.5],
    })
    folds = make_lodo_splits(df, "date")
    print(f"Folds LODO: {len(folds)} (datas únicas).")
    print("Held-out dates:", [f.heldout_date.date().isoformat() for f in folds])
    m = compute_metrics(np.array([1.0,2.0,3.0]), np.array([1.1,1.9,3.2]))
    print("Metrics smoke test:", m)
