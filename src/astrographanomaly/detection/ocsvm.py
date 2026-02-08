from __future__ import annotations
import numpy as np
from sklearn.svm import OneClassSVM

def fit_score_ocsvm(X: np.ndarray, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    # decision_function: higher = inlier -> invert
    anomaly_score = -model.decision_function(X).ravel()
    return model, anomaly_score
