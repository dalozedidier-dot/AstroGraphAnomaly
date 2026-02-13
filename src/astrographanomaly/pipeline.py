from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

from .data.csv import load_csv
from .graph.knn import build_knn_graph
from .features.extract import extract_features
from .detection.isoforest import fit_score_isolation_forest
from .detection.lof import fit_score_lof
from .detection.ocsvm import fit_score_ocsvm
from .detection.robust import score_robust_zscore
from .detection.ensemble import (
    EnsembleConfig,
    fit_score_ensemble,
    parse_engines_csv,
    parse_weights_csv,
)
from .thresholds import ThresholdConfig, label_anomalies
from .reporting.io import write_outputs
from .reporting.plots import save_basic_plots, save_graph_plot
from .utils.manifest import write_manifest
from .llm.prompt_templates import build_prompt

def _fit_score_engine(X_scaled: np.ndarray, engine: str, contamination: float, seed: int) -> np.ndarray:
    """Return anomaly_score with convention: higher = more anomalous."""
    if engine == "isolation_forest":
        _, scores = fit_score_isolation_forest(X_scaled, contamination=contamination, seed=seed)
        return scores
    if engine == "lof":
        # n_neighbors heuristic: ~ sqrt(n)
        nnb = int(max(10, min(50, np.sqrt(len(X_scaled)))))
        _, scores = fit_score_lof(X_scaled, contamination=contamination, n_neighbors=nnb)
        return scores
    if engine == "ocsvm":
        _, scores = fit_score_ocsvm(X_scaled, nu=contamination)
        return scores
    if engine == "robust_zscore":
        return score_robust_zscore(X_scaled)
    if engine == "pineforest":
        from .detection.pineforest import fit_score_pineforest
        _, scores = fit_score_pineforest(X_scaled)
        return scores
    raise ValueError(f"Unknown engine: {engine}")

def run_pipeline(
    mode: str,
    out_dir: str,
    engine: str = "isolation_forest",
    threshold_strategy: str = "contamination",
    contamination: float = 0.05,
    percentile: float = 95.0,
    top_k: int = 50,
    score_threshold: float = 1.0,
    knn_k: int = 8,
    features_mode: str = "extended",
    explain_top: int = 0,
    lime_num_features: int = 8,
    plots: bool = False,
    seed: int = 42,
    # ensemble scoring (engine="ensemble")
    ensemble_engines: str = "isolation_forest,lof,ocsvm",
    ensemble_weights: str = "",
    ensemble_include_graph_constraint: bool = True,
    ensemble_graph_weight: float = 1.5,
    # inputs
    in_csv: Optional[str] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_deg: float = 0.5,
    limit: int = 2000,
) -> Dict[str, Any]:
    """Full workflow. Returns a dict with paths + run metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Ingest
    if mode == "gaia":
        if ra is None or dec is None:
            raise ValueError("mode=gaia requires ra/dec")
        from .data.gaia import fetch_gaia
        df_raw = fetch_gaia(ra=ra, dec=dec, radius_deg=radius_deg, limit=limit)
    elif mode == "csv":
        if in_csv is None:
            raise ValueError("mode=csv requires --in-csv")
        df_raw = load_csv(in_csv)
    elif mode == "hubble":
        if in_csv is None:
            raise ValueError("mode=hubble requires --in-csv")
        from .data.hubble import load_hubble_csv
        df_raw = load_hubble_csv(in_csv)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Minimal schema enforcement
    for col in ["source_id", "ra", "dec"]:
        if col not in df_raw.columns:
            raise ValueError(f"Missing required column: {col}")
    df_raw = df_raw.dropna(subset=["ra", "dec"]).copy()
    df_raw["source_id"] = df_raw["source_id"].astype("int64")

    # 2) Graph
    G = build_knn_graph(df_raw, knn_k=int(knn_k), include_self=False)

    # 3) Features
    X, node_list, feature_names = extract_features(
        G,
        mode=features_mode,
        betweenness_enabled=(features_mode == "extended"),
        betweenness_k=300 if features_mode == "extended" else None,
        betweenness_seed=seed,
        communities_method="louvain",
        articulation_points=(features_mode == "extended"),
        bridges=(features_mode == "extended"),
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) Scores + labels
    per_constraint_raw: Dict[str, np.ndarray] | None = None
    per_constraint_phi: Dict[str, np.ndarray] | None = None
    per_constraint_w: Dict[str, float] | None = None

    if engine == "ensemble":
        engines = parse_engines_csv(ensemble_engines) or ["isolation_forest", "lof", "ocsvm"]
        weights = parse_weights_csv(ensemble_weights)
        cfg = EnsembleConfig(
            engines=engines,
            weights=weights,
            include_graph_constraint=bool(ensemble_include_graph_constraint),
            graph_weight=float(ensemble_graph_weight),
        )
        scores, per_constraint_raw, per_constraint_phi, per_constraint_w = fit_score_ensemble(
            X_scaled=X_scaled,
            X_unscaled=X,
            feature_names=feature_names,
            contamination=float(contamination),
            seed=int(seed),
            cfg=cfg,
        )
    else:
        scores = _fit_score_engine(X_scaled, engine=engine, contamination=contamination, seed=seed)

    thr_cfg = ThresholdConfig(
        strategy=threshold_strategy,
        contamination=contamination,
        percentile=percentile,
        top_k=top_k,
        score=score_threshold,
    )
    labels = label_anomalies(scores, thr_cfg)

    df_scored = pd.DataFrame({"source_id": node_list, "anomaly_score": scores, "anomaly_label": labels})
    if engine == "ensemble" and per_constraint_raw is not None and per_constraint_phi is not None:
        # Keep the contract stable: anomaly_score is the composite score.
        df_scored["incoherence_score"] = df_scored["anomaly_score"]

        # Add per-constraint columns (raw + normalized phi).
        for name, arr in per_constraint_raw.items():
            col = f"score_{name}"
            df_scored[col] = np.asarray(arr, dtype=float)
        for name, arr in per_constraint_phi.items():
            col = f"phi_{name}"
            df_scored[col] = np.asarray(arr, dtype=float)
    df_scored = df_raw.merge(df_scored, on="source_id", how="left")

    # Propagate anomaly fields into graph attributes for GraphML export.
    try:
        score_attr = {int(r["source_id"]): float(r["anomaly_score"]) for _, r in df_scored.iterrows()}
        label_attr = {int(r["source_id"]): int(r["anomaly_label"]) for _, r in df_scored.iterrows()}
        nx.set_node_attributes(G, score_attr, "anomaly_score")
        nx.set_node_attributes(G, label_attr, "anomaly_label")

        if engine == "ensemble" and per_constraint_raw is not None and per_constraint_phi is not None:
            for name, arr in per_constraint_raw.items():
                nx.set_node_attributes(G, {int(sid): float(arr[i]) for i, sid in enumerate(node_list)}, f"score_{name}")
            for name, arr in per_constraint_phi.items():
                nx.set_node_attributes(G, {int(sid): float(arr[i]) for i, sid in enumerate(node_list)}, f"phi_{name}")
    except Exception:
        # GraphML export remains valid even if attribute propagation fails.
        pass

    # Top anomalies (by score desc)
    df_top = df_scored.sort_values("anomaly_score", ascending=False).head(int(top_k)).copy()
    top_nodes = set(df_top["source_id"].astype("int64").tolist())
    G_top = G.subgraph(top_nodes).copy()

    # 5) Exports
    write_outputs(out_dir=out_dir, df_raw=df_raw, df_scored=df_scored, df_top=df_top, G_full=G, G_top=G_top)

    # 6) Explainability + LLM prompts (systematic if explain_top>0)
    explanations_path = None
    prompts_path = None
    if int(explain_top) > 0:
        # detect LIME availability (must have lime.lime_tabular)
        try:
            from lime.lime_tabular import LimeTabularExplainer  # noqa: F401
            from .explain.lime_tabular import explain_with_lime_regression
            lime_available = True
        except Exception:
            explain_with_lime_regression = None
            lime_available = False

        exp_out = out / "explanations.jsonl"
        prm_out = out / "llm_prompts.jsonl"

        df_explain = df_scored.sort_values("anomaly_score", ascending=False).head(int(explain_top))
        idx_map = {sid: i for i, sid in enumerate(node_list)}

        with exp_out.open("w", encoding="utf-8") as fexp, prm_out.open("w", encoding="utf-8") as fprm:
            for _, row in df_explain.iterrows():
                sid = int(row["source_id"])
                idx = idx_map.get(sid)
                if idx is None:
                    continue

                if lime_available and explain_with_lime_regression is not None:
                    lime_payload = explain_with_lime_regression(
                        X_scaled, scores, feature_names,
                        idx=idx,
                        num_features=int(lime_num_features),
                        seed=seed,
                    )
                else:
                    lime_payload = {"idx": int(idx), "local_score": float(scores[idx]), "weights": []}

                snap = {feature_names[j]: float(X_scaled[idx, j]) for j in range(len(feature_names))}

                payload = {
                    "source_id": sid,
                    "anomaly_score": float(row["anomaly_score"]),
                    "engine": engine,
                    "threshold_strategy": threshold_strategy,
                    "feature_snapshot": snap,
                    "lime": lime_payload,
                }

                if engine == "ensemble" and per_constraint_raw is not None and per_constraint_phi is not None and per_constraint_w is not None:
                    constraints: Dict[str, Any] = {}
                    for name, raw_arr in per_constraint_raw.items():
                        phi_arr = per_constraint_phi.get(name)
                        if phi_arr is None:
                            continue
                        constraints[name] = {
                            "weight": float(per_constraint_w.get(name, 1.0)),
                            "raw_score": float(raw_arr[idx]),
                            "phi": float(phi_arr[idx]),
                        }
                    payload["incoherence_score"] = float(scores[idx])
                    payload["constraints"] = constraints
                    payload["ensemble_engines"] = list(parse_engines_csv(ensemble_engines) or [])
                fexp.write(json.dumps(payload, ensure_ascii=False) + "\n")

                prompt_template = "composite" if engine == "ensemble" else "default"
                prompt = build_prompt(payload, template_name=prompt_template)
                fprm.write(json.dumps({"source_id": sid, "prompt": prompt}, ensure_ascii=False) + "\n")

        explanations_path = str(exp_out)
        prompts_path = str(prm_out)

    # 7) Plots
    if plots:
        save_basic_plots(out_dir, df_scored)
        anomalies = set(df_scored.loc[df_scored["anomaly_label"] == -1, "source_id"].astype("int64").tolist())
        save_graph_plot(out_dir, G_top, anomalies=anomalies.intersection(top_nodes))

    # 8) Manifest
    artefacts = {
        "raw": "raw.csv",
        "scored": "scored.csv",
        "top": "top_anomalies.csv",
        "graph_full": "graph_full.graphml",
        "graph_top": "graph_topk.graphml",
    }
    if explanations_path:
        artefacts["explanations"] = "explanations.jsonl"
    if prompts_path:
        artefacts["llm_prompts"] = "llm_prompts.jsonl"
    if plots:
        artefacts["plots_dir"] = "plots/"

    config = dict(
        mode=mode,
        engine=engine,
        threshold=dict(
            strategy=threshold_strategy,
            contamination=contamination,
            percentile=percentile,
            top_k=top_k,
            score=score_threshold,
        ),
        graph=dict(knn_k=knn_k),
        features=dict(mode=features_mode),
        explain=dict(explain_top=explain_top, lime_num_features=lime_num_features),
        plots=plots,
        seed=seed,
        inputs=dict(in_csv=in_csv, ra=ra, dec=dec, radius_deg=radius_deg, limit=limit),
    )

    if engine == "ensemble":
        config["ensemble"] = dict(
            engines=parse_engines_csv(ensemble_engines),
            weights=parse_weights_csv(ensemble_weights),
            include_graph_constraint=bool(ensemble_include_graph_constraint),
            graph_weight=float(ensemble_graph_weight),
        )
    write_manifest(out_dir, config=config, artefacts=artefacts)

    return {
        "out_dir": str(out),
        "artefacts": artefacts,
        "n_rows": int(len(df_raw)),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
    }
