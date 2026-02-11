#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/graph_viz.py

Interactive force-directed graph exploration for AstroGraphAnomaly.

This tool is designed for exploration and showcase:
- force-directed layout
- node color by anomaly score
- rich tooltips with explainability (LIME) when available

Recommended usage (run folder produced by the pipeline):
  python tools/graph_viz.py --run-dir results/<run> --backend plotly --graph topk --dim 2
  python tools/graph_viz.py --run-dir results/<run> --backend pyvis  --graph full --max-nodes 800

Expected in --run-dir:
- scored.csv (required)
- graph_topk.graphml and/or graph_full.graphml (at least one)
- explanations.jsonl (optional; produced when explain_top>0)

Outputs:
- HTML file written to:
  <run_dir>/viz_graph_force/

Dependencies (optional, depending on backend):
- Plotly backend:   pip install "plotly>=5.20"
- PyVis backend:    pip install "pyvis>=0.3"

Notes
- Plotly backend computes positions with NetworkX spring_layout and then exports HTML.
  It is deterministic with --seed.
- PyVis backend uses vis.js physics in the browser; positions keep moving until you
  stabilize/disable physics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def _robust_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    lo = float(np.nanpercentile(x, 5)) if np.any(np.isfinite(x)) else 0.0
    hi = float(np.nanpercentile(x, 95)) if np.any(np.isfinite(x)) else 1.0
    if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
        lo = float(np.nanmin(x)) if np.any(np.isfinite(x)) else 0.0
        hi = float(np.nanmax(x)) if np.any(np.isfinite(x)) else 1.0
        if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
            return np.zeros_like(x, dtype=float)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return y


def _read_scored(scored_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scored_csv)
    if "source_id" not in df.columns:
        raise ValueError("scored.csv must contain a 'source_id' column")

    df = df.copy()
    df["source_id"] = df["source_id"].astype(str)

    if "anomaly_score" in df.columns:
        df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    else:
        df["anomaly_score"] = np.nan

    if "anomaly_label" in df.columns:
        df["anomaly_label"] = pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(1).astype(int)
    else:
        df["anomaly_label"] = 1

    return df


def _read_explanations(explanations_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    """Map source_id -> payload (kept lightweight: only lime weights and metadata)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not explanations_jsonl.exists():
        return out

    with explanations_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = obj.get("source_id")
            if sid is None:
                continue
            sid_s = str(sid)
            lime = obj.get("lime") or {}
            weights = lime.get("weights") or []
            if isinstance(weights, list):
                cleaned: List[Dict[str, Any]] = []
                for w in weights:
                    if not isinstance(w, dict):
                        continue
                    feat = w.get("feature")
                    val = w.get("weight")
                    try:
                        val_f = float(val)
                    except Exception:
                        continue
                    if feat is None:
                        continue
                    cleaned.append({"feature": str(feat), "weight": val_f})
                weights = cleaned

            out[sid_s] = {
                "engine": obj.get("engine"),
                "threshold_strategy": obj.get("threshold_strategy"),
                "lime_weights": weights,
            }

    return out


def _merge_maps(df_scored: pd.DataFrame, lime_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    score_map: Dict[str, float] = {}
    label_map: Dict[str, int] = {}
    weights_map: Dict[str, List[Dict[str, Any]]] = {}

    for _, r in df_scored.iterrows():
        sid = str(r["source_id"])
        sc = r.get("anomaly_score")
        lb = r.get("anomaly_label")

        try:
            sc_f = float(sc)
        except Exception:
            sc_f = float("nan")

        if math.isfinite(sc_f):
            score_map[sid] = sc_f

        try:
            label_map[sid] = int(lb)
        except Exception:
            label_map[sid] = 1

        if sid in lime_map:
            weights = lime_map[sid].get("lime_weights")
            if isinstance(weights, list):
                weights_map[sid] = weights

    return score_map, label_map, weights_map


def _tooltip_html(
    node_id: str,
    score: Optional[float],
    label: Optional[int],
    node_attrs: Dict[str, Any],
    lime_weights: Optional[List[Dict[str, Any]]],
    lime_top: int,
) -> str:
    parts: List[str] = []
    parts.append(f"<b>source_id</b>: {node_id}")

    if score is not None and math.isfinite(score):
        parts.append(f"<b>anomaly_score</b>: {score:.6f}")

    if label is not None:
        parts.append(f"<b>anomaly_label</b>: {label}")

    for c in ["ra", "dec", "distance"]:
        if c in node_attrs:
            try:
                v = float(node_attrs[c])
                if math.isfinite(v):
                    if c in {"ra", "dec"}:
                        parts.append(f"<b>{c}</b>: {v:.6f}")
                    else:
                        parts.append(f"<b>{c}</b>: {v:.3f}")
            except Exception:
                continue

    if lime_weights:
        lw = [w for w in lime_weights if isinstance(w, dict) and "feature" in w and "weight" in w]
        lw = sorted(lw, key=lambda d: abs(float(d.get("weight", 0.0))), reverse=True)
        lw = lw[: max(0, int(lime_top))]
        if lw:
            parts.append("<b>LIME</b> (top contributions):")
            lines = []
            for w in lw:
                feat = str(w.get("feature"))
                try:
                    weight = float(w.get("weight"))
                except Exception:
                    continue
                sign = "+" if weight >= 0 else ""
                lines.append(f"{feat}: {sign}{weight:.4f}")
            if lines:
                parts.append("<br>".join(lines))

    return "<br>".join(parts)


def _select_subgraph(
    G: nx.Graph,
    max_nodes: Optional[int],
    score_map: Dict[str, float],
    seed: int,
) -> nx.Graph:
    if max_nodes is None or max_nodes <= 0 or G.number_of_nodes() <= max_nodes:
        return G

    nodes = [str(n) for n in G.nodes()]

    # Prefer top anomaly scores if available
    scored_nodes = [(n, score_map.get(n)) for n in nodes]
    scored_nodes = [(n, s) for n, s in scored_nodes if s is not None and math.isfinite(float(s))]

    if scored_nodes:
        scored_nodes.sort(key=lambda t: float(t[1]), reverse=True)
        keep = [n for n, _ in scored_nodes[:max_nodes]]
    else:
        rng = random.Random(seed)
        keep = rng.sample(nodes, k=max_nodes)

    return G.subgraph(keep).copy()


def _spring_pos(G: nx.Graph, dim: int, seed: int, iterations: int, k: Optional[float]) -> Dict[str, Tuple[float, ...]]:
    # NetworkX returns a dict keyed by original node ids; normalize to strings.
    raw = nx.spring_layout(G, dim=dim, seed=seed, iterations=iterations, k=k)
    pos: Dict[str, Tuple[float, ...]] = {}
    for n, v in raw.items():
        vv = tuple(float(x) for x in v)
        pos[str(n)] = vv
    return pos


def _write_plotly_html(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, ...]],
    score_map: Dict[str, float],
    label_map: Dict[str, int],
    weights_map: Dict[str, List[Dict[str, Any]]],
    out_html: Path,
    title: str,
    dim: int,
    lime_top: int,
    edge_alpha: float,
    node_size: float,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise SystemExit(
            "Plotly is not installed. Install it with: pip install \"plotly>=5.20\"\n"
            f"Original error: {e}"
        )

    nodes = [str(n) for n in G.nodes() if str(n) in pos]
    if not nodes:
        raise SystemExit("No nodes have positions. Check your graph and layout options.")

    scores = np.array([score_map.get(n, float("nan")) for n in nodes], dtype=float)
    scores = np.where(np.isfinite(scores), scores, np.nan)
    colors = _robust_01(np.nan_to_num(scores, nan=float(np.nanmedian(scores)) if np.any(np.isfinite(scores)) else 0.0))

    sizes = []
    for n in nodes:
        lb = label_map.get(n, 1)
        sizes.append(float(node_size) * (1.8 if lb == -1 else 1.0))

    tooltips = []
    for n in nodes:
        sc = score_map.get(n)
        lb = label_map.get(n, 1)
        attrs = dict(G.nodes[str(n)]) if str(n) in G.nodes else {}
        tooltips.append(_tooltip_html(n, sc, lb, attrs, weights_map.get(n), lime_top=lime_top))

    if dim == 2:
        x = [pos[n][0] for n in nodes]
        y = [pos[n][1] for n in nodes]

        xe: List[Optional[float]] = []
        ye: List[Optional[float]] = []
        for u, v in G.edges():
            su, sv = str(u), str(v)
            if su in pos and sv in pos:
                xe.extend([pos[su][0], pos[sv][0], None])
                ye.extend([pos[su][1], pos[sv][1], None])

        edge_trace = go.Scattergl(
            x=xe,
            y=ye,
            mode="lines",
            line=dict(width=1, color=f"rgba(160,160,160,{max(0.0, min(1.0, edge_alpha))})"),
            hoverinfo="none",
        )

        node_trace = go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=sizes,
                opacity=0.92,
                color=colors,
                colorscale="Viridis",
                colorbar=dict(title="Anomaly score (robust)") if len(nodes) > 10 else None,
            ),
            text=tooltips,
            hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=0, r=0, t=55, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            dragmode="pan",
        )

    else:
        x = [pos[n][0] for n in nodes]
        y = [pos[n][1] for n in nodes]
        z = [pos[n][2] for n in nodes]

        xe: List[Optional[float]] = []
        ye: List[Optional[float]] = []
        ze: List[Optional[float]] = []
        for u, v in G.edges():
            su, sv = str(u), str(v)
            if su in pos and sv in pos:
                xe.extend([pos[su][0], pos[sv][0], None])
                ye.extend([pos[su][1], pos[sv][1], None])
                ze.extend([pos[su][2], pos[sv][2], None])

        edge_trace = go.Scatter3d(
            x=xe,
            y=ye,
            z=ze,
            mode="lines",
            line=dict(width=1, color=f"rgba(160,160,160,{max(0.0, min(1.0, edge_alpha))})"),
            hoverinfo="none",
        )

        node_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=sizes,
                opacity=0.90,
                color=colors,
                colorscale="Viridis",
                colorbar=dict(title="Anomaly score (robust)") if len(nodes) > 10 else None,
            ),
            text=tooltips,
            hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            margin=dict(l=0, r=0, t=55, b=0),
            scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
            dragmode="orbit",
        )

    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def _pyvis_color(score_01: float, alpha: float) -> str:
    """Simple red-green mapping: low score -> green, high score -> red."""
    s = max(0.0, min(1.0, float(score_01)))
    r = int(round(255 * s))
    g = int(round(255 * (1.0 - s)))
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},0,{a:.3f})"


def _write_pyvis_html(
    G: nx.Graph,
    score_map: Dict[str, float],
    label_map: Dict[str, int],
    weights_map: Dict[str, List[Dict[str, Any]]],
    out_html: Path,
    title: str,
    lime_top: int,
    node_size: float,
    physics: bool,
    edge_alpha: float,
) -> None:
    try:
        from pyvis.network import Network
    except Exception as e:
        raise SystemExit(
            "PyVis is not installed. Install it with: pip install \"pyvis>=0.3\"\n"
            f"Original error: {e}"
        )

    nodes = [str(n) for n in G.nodes()]
    scores = np.array([score_map.get(n, float("nan")) for n in nodes], dtype=float)
    scores = np.where(np.isfinite(scores), scores, np.nan)
    fill = float(np.nanmedian(scores)) if np.any(np.isfinite(scores)) else 0.0
    s01 = _robust_01(np.nan_to_num(scores, nan=fill))
    s01_map = {n: float(v) for n, v in zip(nodes, s01.tolist(), strict=False)}

    net = Network(height="820px", width="100%", bgcolor="#0b0f14", font_color="#eaeaea", notebook=False, directed=False)
    net.heading = title

    if physics:
        # Barnes-Hut defaults are generally okay; tweak lightly for stability.
        net.barnes_hut(gravity=-8000, central_gravity=0.35, spring_length=120, spring_strength=0.02, damping=0.20)
    else:
        net.toggle_physics(False)

    # nodes
    for n in nodes:
        sc = score_map.get(n)
        lb = label_map.get(n, 1)
        attrs = dict(G.nodes[str(n)]) if str(n) in G.nodes else {}
        tip = _tooltip_html(n, sc, lb, attrs, weights_map.get(n), lime_top=lime_top)

        size = float(node_size) * (1.8 if lb == -1 else 1.0)
        color = _pyvis_color(s01_map.get(n, 0.0), alpha=0.88)

        net.add_node(
            n,
            label=str(n) if len(nodes) <= 120 else "",
            title=tip,
            color=color,
            size=size,
        )

    # edges
    ea = max(0.0, min(1.0, float(edge_alpha)))
    edge_color = f"rgba(180,180,180,{ea:.3f})"
    for u, v in G.edges():
        net.add_edge(str(u), str(v), color=edge_color)

    # Save
    out_html.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_html))


def _resolve_inputs(run_dir: Optional[Path], graph_choice: str, graph_path: Optional[Path], scored_path: Optional[Path], explanations_path: Optional[Path]) -> Tuple[Path, Path, Optional[Path]]:
    if run_dir is None and (graph_path is None or scored_path is None):
        raise SystemExit("Provide either --run-dir OR both --graph-path and --scored-path")

    if run_dir is not None:
        if scored_path is None:
            scored_path = run_dir / "scored.csv"
        if not scored_path.exists():
            raise SystemExit(f"Missing: {scored_path}")

        if graph_path is None:
            if graph_choice == "topk":
                cand = run_dir / "graph_topk.graphml"
            else:
                cand = run_dir / "graph_full.graphml"
            if not cand.exists():
                # fallback to the other one
                cand2 = run_dir / ("graph_full.graphml" if graph_choice == "topk" else "graph_topk.graphml")
                if cand2.exists():
                    cand = cand2
                else:
                    raise SystemExit(f"Missing graph files in run dir: {run_dir}")
            graph_path = cand

        if explanations_path is None:
            cand_exp = run_dir / "explanations.jsonl"
            if cand_exp.exists():
                explanations_path = cand_exp

    assert graph_path is not None
    assert scored_path is not None

    if not graph_path.exists():
        raise SystemExit(f"Missing: {graph_path}")

    return graph_path, scored_path, explanations_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--run-dir", default=None, help="results/<run> folder (recommended)")
    ap.add_argument("--graph", choices=["topk", "full"], default="topk", help="which graph to load from run dir")

    ap.add_argument("--graph-path", default=None, help="explicit path to a .graphml graph")
    ap.add_argument("--scored-path", default=None, help="explicit path to scored.csv")
    ap.add_argument("--explanations-path", default=None, help="explicit path to explanations.jsonl")

    ap.add_argument("--backend", choices=["plotly", "pyvis"], default="plotly")
    ap.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Plotly spring layout dimension")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--spring-k", type=float, default=None, help="spring_layout k (None uses NetworkX default)")

    ap.add_argument("--max-nodes", type=int, default=0, help="cap nodes (0 = no cap). Keeps top scores if possible")

    ap.add_argument("--lime-top", type=int, default=8, help="how many LIME weights to show in tooltips")
    ap.add_argument("--node-size", type=float, default=6.0)
    ap.add_argument("--edge-alpha", type=float, default=0.18)

    ap.add_argument("--title", default="AstroGraphAnomaly: force-directed graph")
    ap.add_argument("--out", default=None, help="output HTML path. Default: <run_dir>/viz_graph_force/<auto>.html")

    ap.add_argument("--pyvis-no-physics", action="store_true", help="disable PyVis physics (fixed layout)")

    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else None
    graph_path = Path(args.graph_path) if args.graph_path else None
    scored_path = Path(args.scored_path) if args.scored_path else None
    explanations_path = Path(args.explanations_path) if args.explanations_path else None

    graph_path, scored_path, explanations_path = _resolve_inputs(
        run_dir=run_dir,
        graph_choice=str(args.graph),
        graph_path=graph_path,
        scored_path=scored_path,
        explanations_path=explanations_path,
    )

    df_scored = _read_scored(scored_path)
    lime_map = _read_explanations(explanations_path) if explanations_path is not None else {}
    score_map, label_map, weights_map = _merge_maps(df_scored, lime_map)

    G = nx.read_graphml(graph_path)
    # Normalize node ids to strings (GraphML can return non-strings)
    if not all(isinstance(n, str) for n in G.nodes()):
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()}, copy=True)

    max_nodes = int(args.max_nodes) if int(args.max_nodes) > 0 else None
    G2 = _select_subgraph(G, max_nodes=max_nodes, score_map=score_map, seed=int(args.seed))

    out_dir: Path
    if args.out:
        out_html = Path(args.out)
        out_dir = out_html.parent
    else:
        if run_dir is None:
            out_dir = Path.cwd() / "viz_graph_force"
        else:
            out_dir = run_dir / "viz_graph_force"

        suffix = f"{args.backend}_{args.graph}_dim{args.dim}.html" if args.backend == "plotly" else f"{args.backend}_{args.graph}.html"
        out_html = out_dir / suffix

    out_dir.mkdir(parents=True, exist_ok=True)

    title = str(args.title)
    if max_nodes is not None:
        title = f"{title} (n={G2.number_of_nodes()}, m={G2.number_of_edges()})"

    if args.backend == "plotly":
        pos = _spring_pos(G2, dim=int(args.dim), seed=int(args.seed), iterations=int(args.iterations), k=args.spring_k)
        _write_plotly_html(
            G=G2,
            pos=pos,
            score_map=score_map,
            label_map=label_map,
            weights_map=weights_map,
            out_html=out_html,
            title=title,
            dim=int(args.dim),
            lime_top=int(args.lime_top),
            edge_alpha=float(args.edge_alpha),
            node_size=float(args.node_size),
        )

    else:
        _write_pyvis_html(
            G=G2,
            score_map=score_map,
            label_map=label_map,
            weights_map=weights_map,
            out_html=out_html,
            title=title,
            lime_top=int(args.lime_top),
            node_size=float(args.node_size),
            physics=(not bool(args.pyvis_no_physics)),
            edge_alpha=float(args.edge_alpha),
        )

    print(f"[graph_viz] wrote: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
