from __future__ import annotations
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def fit_score_lof(X: np.ndarray, contamination: float = 0.05, n_neighbors: int = 35):
    # novelty=True for scoring on training data via decision_function; but here we fit+score on same X
    model = LocalOutlierFactor(n_neighbors=min(int(n_neighbors), max(2, len(X)-1)), contamination=contamination, novelty=False)
    labels = model.fit_predict(X)
    # negative_outlier_factor_ is negative; more negative = more anomalous -> invert sign
    anomaly_score = -model.negative_outlier_factor_
    return model, anomaly_score
