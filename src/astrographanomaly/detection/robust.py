from __future__ import annotations
import numpy as np

def robust_zscore(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + eps
    z = 0.6745 * (X - med) / mad
    return z

def score_robust_zscore(X: np.ndarray) -> np.ndarray:
    z = robust_zscore(X)
    # aggregate magnitude
    return np.sqrt((z**2).sum(axis=1))
