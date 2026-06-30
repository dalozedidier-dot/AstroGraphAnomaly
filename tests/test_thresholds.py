"""Unit tests for thresholding and ensemble parsing helpers."""

from __future__ import annotations

import numpy as np

from astrographanomaly.thresholds import ThresholdConfig, label_anomalies
from astrographanomaly.detection.ensemble import (
    parse_engines_csv,
    parse_weights_csv,
    robust_minmax_01,
)


def test_label_top_k_marks_exactly_k():
    scores = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    labels = label_anomalies(scores, ThresholdConfig(strategy="top_k", top_k=2))
    assert int((labels == -1).sum()) == 2
    # Highest two scores (idx 1 and 4) must be the anomalies.
    assert labels[1] == -1 and labels[4] == -1


def test_label_score_threshold():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    labels = label_anomalies(scores, ThresholdConfig(strategy="score", score=1.5))
    assert list(labels) == [1, 1, -1, -1]


def test_label_percentile():
    scores = np.arange(100, dtype=float)
    labels = label_anomalies(scores, ThresholdConfig(strategy="percentile", percentile=95.0))
    # Top ~5% flagged.
    assert int((labels == -1).sum()) >= 5


def test_label_empty_input():
    labels = label_anomalies(np.array([]), ThresholdConfig())
    assert labels.size == 0


def test_parse_engines_csv():
    assert parse_engines_csv("isolation_forest, lof ,ocsvm") == [
        "isolation_forest",
        "lof",
        "ocsvm",
    ]
    assert parse_engines_csv("") == []


def test_parse_weights_csv_ignores_invalid():
    out = parse_weights_csv("lof=2,ocsvm=abc,iso=-1,graph=1.5,bad")
    assert out == {"lof": 2.0, "graph": 1.5}


def test_robust_minmax_constant_input():
    # Degenerate (no spread) input must not produce NaNs.
    out = robust_minmax_01(np.full(10, 3.0))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))
