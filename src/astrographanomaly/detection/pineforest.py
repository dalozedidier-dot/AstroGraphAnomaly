from __future__ import annotations
import numpy as np

def fit_score_pineforest(X: np.ndarray):
    from coniferest.pineforest import PineForest
    model = PineForest()
    model.fit(X)
    scores = model.anomaly_score(X)  # higher = more anomalous
    return model, np.asarray(scores, dtype=float)
