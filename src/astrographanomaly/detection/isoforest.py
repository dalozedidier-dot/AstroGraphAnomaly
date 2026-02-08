from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest

def fit_score_isolation_forest(X: np.ndarray, contamination: float = 0.05, seed: int = 42):
    model = IsolationForest(contamination=contamination, random_state=seed)
    model.fit(X)
    # sklearn: higher score_samples = more normal -> invert
    anomaly_score = -model.score_samples(X)
    return model, anomaly_score
