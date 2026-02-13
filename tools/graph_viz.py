#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — Graph Explorer (force-directed, interactive)
---------------------------------------------------------------

Exports an interactive force-directed visualization from a NetworkX GraphML graph
plus a scored.csv file. Supports:
- Plotly (2D or 3D) -> standalone HTML (Plotly JS embedded, works offline)
- PyVis (vis.js physics) -> standalone HTML

Coloring:
- By anomaly score (default)
- Tooltips include explainability features when `explanations.jsonl` is provided.

Typical usage:
  python tools/graph_viz.py --run-dir results/<run> --backend plotly --dim 2
  python tools/graph_viz.py --run-dir results/<run> --backend pyvis  --max-nodes 1200

Inputs expected in --run-dir (defaults):
- scored.csv (required)
- graph_full.graphml or graph_topk.graphml (required)
- explanations.jsonl (optional)

Outputs:
- results/<run>/viz_graph_force/plotly_topk_dim2.html, etc.

Dependencies:
- plotly (for Plotly backend):   pip install -r requirements_viz.txt
- pyvis  (for PyVis backend):    pip install -r requirements_viz.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def _robust_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=float)
    lo = float(np.nanpercentile(x, 5))
    hi = float(np.nanpercentile(x, 95))
    if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
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


def _read_explanations(explanations_jsonl: Optional[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """Return source_id -> list of {feature, weight}."""
    if explanations_jsonl is None or not explanations_jsonl.exists():
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
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
            if not isinstance(weights, list):
                continue
            cleaned: List[Dict[str, Any]] = []
            for w in weights:
                if not isinstance(w, dict):
                    continue
                feat = w.get("feature")
                val = w.get("weight")
                if feat is None:
                    continue
                try:
                    val_f = float(val)
                except Exception:
                    continue
                cleaned.append({"feature": str(feat), "weight": float(val_f)})
            if cleaned:
                out[sid_s] = cleaned
    return out


def _merge_maps(
    df_scored: pd.DataFrame,
    lime_map: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    score_map: Dict[str, float] = {}
    label_map: Dict[str, int] = {}
    for _, r in df_scored.iterrows():
        sid = str(r["source_id"])
        sc = r.get("anomaly_score", np.nan)
        lb = r.get("anomaly_label", 1)
        try:
            sc_f = float(sc) if sc is not None and math.isfinite(float(sc)) else 0.0
        except Exception:
            sc_f = 0.0
        try:
            lb_i = int(lb)
        except Exception:
            lb_i = 1
        score_map[sid] = sc_f
        label_map[sid] = lb_i

    return score_map, label_map, lime_map


def _resolve_inputs(
    run_dir: Optional[Path],
    graph_choice: str,
    graph_path: Optional[Path],
    scored_path: Optional[Path],
    explanations_path: Optional[Path],
) -> Tuple[Path, Path, Optional[Path]]:
    if run_dir is None and (graph_path is None or scored_path is None):
        raise ValueError("Provide --run-dir or both --graph-path and --scored-path")

    if scored_path is None:
        scored_path = run_dir / "scored.csv"  # type: ignore[operator]
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored.csv: {scored_path}")

    if graph_path is None:
        if run_dir is None:
            raise ValueError("Provide --graph-path if --run-dir is not set")
        if graph_choice == "topk":
            for nm in ["graph_topk.graphml", "graph_union.graphml", "graph_full.graphml"]:
                p = run_dir / nm
                if p.exists():
                    graph_path = p
                    break
        else:
            for nm in ["graph_full.graphml", "graph_union.graphml", "graph_topk.graphml"]:
                p = run_dir / nm
                if p.exists():
                    graph_path = p
                    break
    if graph_path is None or not graph_path.exists():
        raise FileNotFoundError("Missing graph file (graph_full.graphml / graph_topk.graphml).")

    if explanations_path is None and run_dir is not None:
        exp = run_dir / "explanations.jsonl"
        explanations_path = exp if exp.exists() else None

    return graph_path, scored_path, explanations_path


def _pick_top_nodes_by_score(nodes: List[str], score_map: Dict[str, float], k: int) -> List[str]:
    scored = [(n, float(score_map.get(n, 0.0))) for n in nodes]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [n for n, _ in scored[:k]]


def _select_subgraph(
    G: nx.Graph,
    max_nodes: Optional[int],
    score_map: Dict[str, float],
    seed: int,
) -> nx.Graph:
    if max_nodes is None or G.number_of_nodes() <= max_nodes:
        return G

    nodes = [str(n) for n in G.nodes()]
    top = _pick_top_nodes_by_score(nodes, score_map, max_nodes)
    keep = set(top)
    # keep neighbors for context
    for n in top:
        for nb in G.neighbors(n):
            keep.add(str(nb))
    H = G.subgraph(list(keep)).copy()

    if H.number_of_nodes() > max_nodes:
        top2 = _pick_top_nodes_by_score(list(H.nodes()), score_map, max_nodes)
        keep2 = set(top2)
        rng = random.Random(seed)
        rest = [n for n in H.nodes() if n not in keep2]
        rng.shuffle(rest)
        for n in rest:
            if len(keep2) >= max_nodes:
                break
            keep2.add(n)
        H = H.subgraph(list(keep2)).copy()

    return H


def _fmt_float(x: Any, nd: int = 4) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return str(x)
    return f"{xf:.{nd}f}"


def _tooltip_html(
    node_id: str,
    score_map: Dict[str, float],
    label_map: Dict[str, int],
    weights_map: Dict[str, List[Dict[str, Any]]],
    node_attrs: Dict[str, Any],
    lime_top: int,
) -> str:
    sc = score_map.get(node_id, 0.0)
    lb = label_map.get(node_id, 1)
    parts: List[str] = []
    parts.append(f"<b>source_id</b>: {node_id}")
    parts.append(f"<b>anomaly_score</b>: {_fmt_float(sc, 6)}")
    parts.append(f"<b>anomaly_label</b>: {lb}")

    # If enriched columns exist on the node (GraphML), include a couple
    for k in ["ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp", "ruwe"]:
        if k in node_attrs:
            parts.append(f"<b>{k}</b>: {_fmt_float(node_attrs.get(k), 4)}")

    lw = weights_map.get(node_id)
    if lw:
        lw2 = sorted(lw, key=lambda w: abs(float(w.get("weight", 0.0))), reverse=True)[: max(1, int(lime_top))]
        items = []
        for w in lw2:
            feat = str(w.get("feature"))
            ww = w.get("weight")
            items.append(f"<li><code>{feat}</code> : {_fmt_float(ww, 5)}</li>")
        parts.append("<b>LIME (top)</b>:<br><ul>" + "".join(items) + "</ul>")
    else:
        parts.append("<i>No LIME payload</i>")

    return "<br>".join(parts)


def _spring_pos(G: nx.Graph, dim: int, seed: int, iterations: int, k: Optional[float]) -> Dict[str, np.ndarray]:
    if k is None:
        n = max(1, G.number_of_nodes())
        k = 1.0 / math.sqrt(float(n))
    raw = nx.spring_layout(G, dim=dim, seed=seed, iterations=int(iterations), k=float(k))
    pos: Dict[str, np.ndarray] = {}
    for n, v in raw.items():
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size != dim:
            continue
        pos[str(n)] = arr
    return pos


def _plotly_write_html(fig, out_html: Path) -> None:
    # include_plotlyjs=True embeds plotly.js (works offline)
    try:
        from plotly.offline import plot as plotly_plot  # type: ignore
    except Exception as e:
        raise SystemExit('Plotly not installed. Install with: pip install -r requirements_viz.txt') from e

    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs=True)


def _plotly_force_2d(
    G: nx.Graph,
    pos: Dict[str, np.ndarray],
    score_map: Dict[str, float],
    label_map: Dict[str, int],
    weights_map: Dict[str, List[Dict[str, Any]]],
    title: str,
    lime_top: int,
    edge_alpha: float,
    node_size: float,
):
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        raise SystemExit('Plotly not installed. Install with: pip install -r requirements_viz.txt') from e

    nodes = [str(n) for n in G.nodes() if str(n) in pos]
    scores = np.array([float(score_map.get(n, 0.0)) for n in nodes], dtype=float)
    s01 = _robust_01(scores)
    s01_map = {n: float(v) for n, v in zip(nodes, s01.tolist(), strict=False)}

    xe: List[Optional[float]] = []
    ye: List[Optional[float]] = []
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su not in pos or sv not in pos:
            continue
        xe.extend([float(pos[su][0]), float(pos[sv][0]), None])
        ye.extend([float(pos[su][1]), float(pos[sv][1]), None])

    edge_trace = go.Scattergl(
        x=xe,
        y=ye,
        mode="lines",
        line=dict(width=1, color=f"rgba(160,160,160,{max(0.0, min(1.0, float(edge_alpha)))})"),
        hoverinfo="none",
        showlegend=False,
    )

    x = [float(pos[n][0]) for n in nodes]
    y = [float(pos[n][1]) for n in nodes]
    sizes = []
    texts = []
    for n in nodes:
        lb = int(label_map.get(n, 1))
        sizes.append(float(node_size) * (1.45 if lb == -1 else 1.0))
        texts.append(_tooltip_html(n, score_map, label_map, weights_map, dict(G.nodes[n]), lime_top))

    node_trace = go.Scattergl(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=sizes,
            color=[s01_map.get(n, 0.0) for n in nodes],
            colorscale="Viridis",
            opacity=0.90,
            colorbar=dict(title="Anomaly score (robust 0..1)"),
        ),
        text=texts,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        height=900,
        margin=dict(l=0, r=0, t=55, b=0),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _plotly_force_3d(
    G: nx.Graph,
    pos: Dict[str, np.ndarray],
    score_map: Dict[str, float],
    label_map: Dict[str, int],
    weights_map: Dict[str, List[Dict[str, Any]]],
    title: str,
    lime_top: int,
    edge_alpha: float,
    node_size: float,
):
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        raise SystemExit('Plotly not installed. Install with: pip install -r requirements_viz.txt') from e

    nodes = [str(n) for n in G.nodes() if str(n) in pos]
    scores = np.array([float(score_map.get(n, 0.0)) for n in nodes], dtype=float)
    s01 = _robust_01(scores)
    s01_map = {n: float(v) for n, v in zip(nodes, s01.tolist(), strict=False)}

    xe: List[Optional[float]] = []
    ye: List[Optional[float]] = []
    ze: List[Optional[float]] = []
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su not in pos or sv not in pos:
            continue
        xe.extend([float(pos[su][0]), float(pos[sv][0]), None])
        ye.extend([float(pos[su][1]), float(pos[sv][1]), None])
        ze.extend([float(pos[su][2]), float(pos[sv][2]), None])

    edge_trace = go.Scatter3d(
        x=xe,
        y=ye,
        z=ze,
        mode="lines",
        line=dict(width=1, color=f"rgba(160,160,160,{max(0.0, min(1.0, float(edge_alpha)))})"),
        hoverinfo="none",
        showlegend=False,
    )

    x = [float(pos[n][0]) for n in nodes]
    y = [float(pos[n][1]) for n in nodes]
    z = [float(pos[n][2]) for n in nodes]
    sizes = []
    texts = []
    for n in nodes:
        lb = int(label_map.get(n, 1))
        sizes.append(float(node_size) * (1.45 if lb == -1 else 1.0))
        texts.append(_tooltip_html(n, score_map, label_map, weights_map, dict(G.nodes[n]), lime_top))

    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=sizes,
            color=[s01_map.get(n, 0.0) for n in nodes],
            colorscale="Viridis",
            opacity=0.90,
            colorbar=dict(title="Anomaly score (robust 0..1)"),
        ),
        text=texts,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        height=900,
        dragmode="orbit",
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.15)),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        showlegend=False,
    )
    return fig


def _rgba_redscale(v01: float, alpha: float = 0.85) -> str:
    v = float(max(0.0, min(1.0, v01)))
    r = 255
    g = int(230 * (1.0 - v) + 25 * v)
    b = int(60 * (1.0 - v))
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a:.3f})"


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
        from pyvis.network import Network  # type: ignore
    except Exception as e:
        raise SystemExit('PyVis not installed. Install with: pip install -r requirements_viz.txt') from e

    nodes = [str(n) for n in G.nodes()]
    scores = np.array([float(score_map.get(n, 0.0)) for n in nodes], dtype=float)
    s01 = _robust_01(scores)
    s01_map = {n: float(v) for n, v in zip(nodes, s01.tolist(), strict=False)}

    net = Network(height="820px", width="100%", bgcolor="#0b0b10", font_color="#f2f2f2", directed=False)
    net.heading = title

    if physics:
        net.barnes_hut(gravity=-80000, central_gravity=0.22, spring_length=110, spring_strength=0.010, damping=0.11)

    for n in nodes:
        lb = int(label_map.get(n, 1))
        size = float(node_size) * (1.45 if lb == -1 else 1.0)
        color = _rgba_redscale(s01_map.get(n, 0.0), alpha=0.88)
        title_html = _tooltip_html(n, score_map, label_map, weights_map, dict(G.nodes[n]), lime_top)
        net.add_node(n, label=str(n), title=title_html, color=color, size=size)

    edge_color = f"rgba(160,160,160,{max(0.0, min(1.0, float(edge_alpha)))})"
    for u, v in G.edges():
        net.add_edge(str(u), str(v), color=edge_color)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    net.show(str(out_html))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="graph_viz.py",
        description="AstroGraphAnomaly: interactive force-directed graph visualization (Plotly/PyVis).",
    )

    ap.add_argument("--run-dir", default=None, help="run folder containing scored.csv + graph_*.graphml")
    ap.add_argument("--graph", choices=["topk", "full"], default="topk")
    ap.add_argument("--graph-path", default=None, help="explicit graphml path (overrides --run-dir/--graph)")
    ap.add_argument("--scored-path", default=None, help="explicit scored.csv path (overrides --run-dir)")
    ap.add_argument("--explanations-path", default=None, help="explicit explanations.jsonl path (optional)")

    ap.add_argument("--backend", choices=["plotly", "pyvis"], default="plotly")
    ap.add_argument("--dim", choices=[2, 3], type=int, default=2, help="Plotly only: 2D or 3D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iterations", type=int, default=250)
    ap.add_argument("--spring-k", type=float, default=None, help="spring_layout k (default: 1/sqrt(n))")

    ap.add_argument("--max-nodes", type=int, default=0, help="cap nodes (0 disables). keeps top anomalies + neighbors.")
    ap.add_argument("--lime-top", type=int, default=6, help="how many LIME features to show in tooltip")

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
    weights_map = _read_explanations(explanations_path)
    score_map, label_map, weights_map = _merge_maps(df_scored, weights_map)

    G = nx.read_graphml(graph_path)
    if not all(isinstance(n, str) for n in G.nodes()):
        G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()}, copy=True)

    max_nodes = int(args.max_nodes) if int(args.max_nodes) > 0 else None
    G2 = _select_subgraph(G, max_nodes=max_nodes, score_map=score_map, seed=int(args.seed))

    if args.out:
        out_html = Path(args.out)
    else:
        out_dir = (run_dir / "viz_graph_force") if run_dir is not None else (Path.cwd() / "viz_graph_force")
        suffix = f"{args.backend}_{args.graph}_dim{args.dim}.html" if args.backend == "plotly" else f"{args.backend}_{args.graph}.html"
        out_html = out_dir / suffix

    out_html.parent.mkdir(parents=True, exist_ok=True)

    title = str(args.title)
    if max_nodes is not None:
        title = f"{title} (n={G2.number_of_nodes()}, m={G2.number_of_edges()})"

    if args.backend == "plotly":
        pos = _spring_pos(G2, dim=int(args.dim), seed=int(args.seed), iterations=int(args.iterations), k=args.spring_k)
        if int(args.dim) == 2:
            fig = _plotly_force_2d(
                G=G2,
                pos=pos,
                score_map=score_map,
                label_map=label_map,
                weights_map=weights_map,
                title=title,
                lime_top=int(args.lime_top),
                edge_alpha=float(args.edge_alpha),
                node_size=float(args.node_size),
            )
        else:
            fig = _plotly_force_3d(
                G=G2,
                pos=pos,
                score_map=score_map,
                label_map=label_map,
                weights_map=weights_map,
                title=title,
                lime_top=int(args.lime_top),
                edge_alpha=float(args.edge_alpha),
                node_size=float(args.node_size),
            )
        _plotly_write_html(fig, out_html)
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
