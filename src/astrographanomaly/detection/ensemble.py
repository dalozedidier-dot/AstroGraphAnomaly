from __future__ import annotations

"""Ensemble (constraint-style) scoring for AstroGraphAnomaly.

Goal
  Combine multiple anomaly engines into one composite score that is:
  - robust to scale differences between engines
  - easy to sweep (weights, engine sets)
  - compatible with the existing thresholding + export contracts

Model
  Treat each engine (and optionally a graph constraint) as a "constraint" i with:
    raw_i(x)  : engine score (higher = more anomalous)
    phi_i(x)  : normalized violation in [0, 1]
    w_i       : weight

  Composite (a.k.a. incoherence) score:
    S(x) = sum_i w_i * phi_i(x) / sum_i w_i

Notes
  - Normalization uses a robust percentile min-max (5th..95th by default).
  - The optional graph constraint is derived from graph metrics already present
    in the extracted feature matrix (betweenness, articulation, bridge incidence).
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .isoforest import fit_score_isolation_forest
from .lof import fit_score_lof
from .ocsvm import fit_score_ocsvm
from .robust import score_robust_zscore


@dataclass(frozen=True)
class EnsembleConfig:
    engines: List[str]
    weights: Dict[str, float]
    include_graph_constraint: bool = True
    graph_weight: float = 1.5
    normalize_lo: float = 5.0
    normalize_hi: float = 95.0


def parse_engines_csv(s: str) -> List[str]:
    engines = [x.strip() for x in (s or "").split(",") if x.strip()]
    return engines


def parse_weights_csv(s: str) -> Dict[str, float]:
    """Parse 'a=1,b=0.5' into dict. Ignores invalid entries."""
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv) or fv <= 0:
            continue
        out[k] = float(fv)
    return out


def robust_minmax_01(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=float)

    p_lo = float(np.nanpercentile(x, lo))
    p_hi = float(np.nanpercentile(x, hi))
    if not (math.isfinite(p_lo) and math.isfinite(p_hi)) or (p_hi - p_lo) < 1e-12:
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        if not (math.isfinite(mn) and math.isfinite(mx)) or (mx - mn) < 1e-12:
            return np.zeros_like(x, dtype=float)
        p_lo, p_hi = mn, mx

    y = (x - p_lo) / (p_hi - p_lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return y


def _fit_engine_scores(
    X_scaled: np.ndarray,
    engine: str,
    contamination: float,
    seed: int,
) -> np.ndarray:
    """Return anomaly_score with convention: higher = more anomalous."""
    if engine == "isolation_forest":
        _, scores = fit_score_isolation_forest(X_scaled, contamination=contamination, seed=seed)
        return scores
    if engine == "lof":
        nnb = int(max(10, min(50, np.sqrt(len(X_scaled)))))
        _, scores = fit_score_lof(X_scaled, contamination=contamination, n_neighbors=nnb)
        return scores
    if engine == "ocsvm":
        _, scores = fit_score_ocsvm(X_scaled, nu=contamination)
        return scores
    if engine == "robust_zscore":
        return score_robust_zscore(X_scaled)
    if engine == "pineforest":
        try:
            from .pineforest import fit_score_pineforest
        except Exception as e:
            raise RuntimeError("pineforest engine not available (missing coniferest)") from e
        _, scores = fit_score_pineforest(X_scaled)
        return scores
    raise ValueError(f"Unknown engine: {engine}")


def _graph_constraint_raw(X_unscaled: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Graph-derived violation score from extracted features.

    Uses only features that exist in extended mode. If missing, returns zeros.
    """

    fn = {name: i for i, name in enumerate(feature_names)}
    n = X_unscaled.shape[0]
    if n == 0:
        return np.asarray([], dtype=float)

    if "betweenness" not in fn:
        return np.zeros(n, dtype=float)

    btw = np.asarray(X_unscaled[:, fn["betweenness"]], dtype=float)

    is_art = np.zeros(n, dtype=float)
    if "is_articulation" in fn:
        is_art = np.asarray(X_unscaled[:, fn["is_articulation"]], dtype=float)

    inc_bridge = np.zeros(n, dtype=float)
    if "incident_to_bridge" in fn:
        inc_bridge = np.asarray(X_unscaled[:, fn["incident_to_bridge"]], dtype=float)

    # Simple, explainable recipe: high betweenness + structural "choke points".
    raw = (1.0 * btw) + (0.70 * is_art) + (0.40 * inc_bridge)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    return raw


def fit_score_ensemble(
    X_scaled: np.ndarray,
    X_unscaled: np.ndarray,
    feature_names: List[str],
    contamination: float,
    seed: int,
    cfg: EnsembleConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Compute composite score + per-constraint components.

    Returns:
      composite_scores: (n,)
      raw_scores: dict[name] -> (n,)
      phi_scores: dict[name] -> (n,) normalized 0..1
      weights_used: dict[name] -> float
    """

    n = int(X_scaled.shape[0])
    if n == 0:
        z = np.asarray([], dtype=float)
        return z, {}, {}, {}

    raw_scores: Dict[str, np.ndarray] = {}
    phi_scores: Dict[str, np.ndarray] = {}

    engines = [e.strip() for e in (cfg.engines or []) if e.strip()]
    if not engines:
        engines = ["isolation_forest", "lof", "ocsvm"]

    # Fit each engine, skipping those that are unavailable.
    usable: List[str] = []
    for eng in engines:
        try:
            raw = _fit_engine_scores(X_scaled, eng, contamination=contamination, seed=seed)
        except Exception:
            continue
        raw = np.asarray(raw, dtype=float).reshape(-1)
        if raw.size != n:
            continue
        raw_scores[eng] = raw
        phi_scores[eng] = robust_minmax_01(raw, lo=cfg.normalize_lo, hi=cfg.normalize_hi)
        usable.append(eng)

    if cfg.include_graph_constraint:
        graw = _graph_constraint_raw(X_unscaled, feature_names)
        if graw.size == n:
            raw_scores["graph"] = graw
            phi_scores["graph"] = robust_minmax_01(graw, lo=cfg.normalize_lo, hi=cfg.normalize_hi)

    if not raw_scores:
        raise RuntimeError("Ensemble has no usable constraints (all engines failed)")

    # Resolve weights: start from defaults, then override with cfg.weights.
    weights_used: Dict[str, float] = {}
    for name in raw_scores.keys():
        if name == "graph":
            weights_used[name] = float(cfg.graph_weight)
        else:
            weights_used[name] = float(cfg.weights.get(name, 1.0))

    # Drop non-positive weights defensively.
    weights_used = {k: v for k, v in weights_used.items() if math.isfinite(v) and v > 0}
    if not weights_used:
        raise RuntimeError("Ensemble weights invalid (no positive weights)")

    denom = float(sum(weights_used.values()))
    comp = np.zeros(n, dtype=float)
    for name, w in weights_used.items():
        phi = phi_scores.get(name)
        if phi is None:
            continue
        comp += float(w) * np.asarray(phi, dtype=float)

    comp = comp / max(1e-12, denom)
    comp = np.where(np.isfinite(comp), comp, 0.0)
    return comp, raw_scores, phi_scores, weights_used
