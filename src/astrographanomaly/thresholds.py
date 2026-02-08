from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class ThresholdConfig:
    strategy: str = "contamination"  # contamination | percentile | top_k | score
    contamination: float = 0.05
    percentile: float = 95.0
    top_k: int = 50
    score: float = 1.0

def label_anomalies(scores: np.ndarray, cfg: ThresholdConfig) -> np.ndarray:
    """Return labels: -1 anomaly, +1 normal. Convention: higher score = more anomalous."""
    n = len(scores)
    if n == 0:
        return np.array([], dtype=int)

    s = np.asarray(scores, dtype=float)

    if cfg.strategy == "score":
        thr = float(cfg.score)
        return np.where(s >= thr, -1, 1)

    if cfg.strategy == "top_k":
        k = max(1, min(int(cfg.top_k), n))
        idx = np.argsort(-s)[:k]
        labels = np.ones(n, dtype=int)
        labels[idx] = -1
        return labels

    if cfg.strategy == "percentile":
        p = float(cfg.percentile)
        thr = np.percentile(s, p)
        return np.where(s >= thr, -1, 1)

    # contamination default
    c = float(cfg.contamination)
    k = max(1, min(int(round(c * n)), n))
    idx = np.argsort(-s)[:k]
    labels = np.ones(n, dtype=int)
    labels[idx] = -1
    return labels
