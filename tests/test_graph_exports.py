"""Regression tests for the GraphML export contract.

Graph node IDs are strings (see ``build_knn_graph``). The pipeline must
stringify source_ids when selecting the top-k subgraph and when propagating
anomaly attributes, otherwise ``graph_topk.graphml`` comes out empty and
``graph_full.graphml`` carries no anomaly fields.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from astrographanomaly.pipeline import run_pipeline


def test_graphml_exports_are_populated(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    top_k = 15
    run_pipeline(
        mode="csv",
        in_csv="data/sample_gaia_like.csv",
        out_dir=str(out_dir),
        engine="robust_zscore",
        threshold_strategy="top_k",
        top_k=top_k,
        knn_k=8,
        features_mode="extended",
        explain_top=0,
        plots=False,
        seed=42,
    )

    g_top = nx.read_graphml(out_dir / "graph_topk.graphml")
    assert g_top.number_of_nodes() == top_k

    g_full = nx.read_graphml(out_dir / "graph_full.graphml")
    assert g_full.number_of_nodes() > 0
    # Every node should carry the propagated anomaly fields.
    for _, data in g_full.nodes(data=True):
        assert "anomaly_score" in data
        assert "anomaly_label" in data


def test_ensemble_propagates_per_constraint_attrs(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_ens"
    run_pipeline(
        mode="csv",
        in_csv="data/sample_gaia_like.csv",
        out_dir=str(out_dir),
        engine="ensemble",
        threshold_strategy="top_k",
        top_k=10,
        knn_k=8,
        features_mode="extended",
        explain_top=0,
        plots=False,
        seed=42,
    )

    g_full = nx.read_graphml(out_dir / "graph_full.graphml")
    _, data = next(iter(g_full.nodes(data=True)))
    assert any(k.startswith("score_") for k in data)
    assert any(k.startswith("phi_") for k in data)
