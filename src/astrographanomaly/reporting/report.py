"""Human-readable run report for the core pipeline.

Every run already emits CSV/GraphML/manifest artefacts. This module adds two
high-value, easy-to-consume outputs:

- ``summary.json`` : compact run statistics (counts, score distribution, graph
  stats, embedding mode, top anomalies) — convenient for dashboards/CI assertions.
- ``report.html``  : a *self-contained* page (plots embedded as base64) with a
  stats header, the top-anomalies table, and the raw summary — openable offline,
  no relative asset paths required.

Design goals: zero new dependencies, robust to missing columns/plots, and stable
output ordering so reports diff cleanly between runs.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

# Astro columns surfaced (when present) in the top-anomalies table / summary.
_ASTRO_COLS: List[str] = [
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "bp_rp",
    "ruwe",
]

# Plot file -> human title, in display order.
_PLOT_TITLES: Dict[str, str] = {
    "score_hist.png": "Distribution des scores d'anomalie",
    "ra_dec_score.png": "Carte spatiale (RA/Dec) colorée par score",
    "cmd_bp_rp_vs_g.png": "Diagramme Couleur–Magnitude (BP-RP vs G)",
    "mag_vs_distance.png": "Magnitude vs distance",
    "mean_features_anom_vs_normal.png": "Moyennes de features : anomal vs normal",
    "top_anomalies_scores.png": "Top anomalies (score par source_id)",
    "pca_2d.png": "Projection PCA (2D)",
    "graph_communities_anomalies.png": "Graphe : communautés + anomalies",
}


def _score_stats(scores: np.ndarray) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {}
    pct = np.percentile(s, [50, 90, 99])
    return {
        "min": float(s.min()),
        "p50": float(pct[0]),
        "p90": float(pct[1]),
        "p99": float(pct[2]),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def _top_records(df_top: pd.DataFrame, limit: int = 25) -> List[Dict[str, Any]]:
    cols = ["source_id", "anomaly_score"] + [c for c in _ASTRO_COLS if c in df_top.columns]
    recs: List[Dict[str, Any]] = []
    for _, r in df_top.head(limit).iterrows():
        rec: Dict[str, Any] = {}
        for c in cols:
            v = r.get(c)
            if c == "source_id":
                rec[c] = int(v)
            else:
                try:
                    fv = float(v)
                    rec[c] = fv if np.isfinite(fv) else None
                except (TypeError, ValueError):
                    rec[c] = None
        recs.append(rec)
    return recs


def build_summary(
    df_scored: pd.DataFrame,
    df_top: pd.DataFrame,
    G: nx.Graph,
    config: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the run summary dict (also serialized to summary.json)."""
    n_rows = int(len(df_scored))
    n_anom = int((df_scored.get("anomaly_label") == -1).sum()) if "anomaly_label" in df_scored else 0
    scores = df_scored["anomaly_score"].to_numpy() if "anomaly_score" in df_scored else np.array([])

    return {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "engine": config.get("engine"),
        "threshold": config.get("threshold"),
        "features_mode": (config.get("features") or {}).get("mode"),
        "seed": config.get("seed"),
        "embedding_mode": G.graph.get("embedding_mode"),
        "counts": {
            "n_rows": n_rows,
            "n_nodes": int(run_meta.get("n_nodes", G.number_of_nodes())),
            "n_edges": int(run_meta.get("n_edges", G.number_of_edges())),
            "n_anomalies": n_anom,
            "anomaly_fraction": float(n_anom / n_rows) if n_rows else 0.0,
        },
        "score_stats": _score_stats(scores),
        "top_anomalies": _top_records(df_top),
    }


def _img_tag(path: Path, title: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure><figcaption>{html.escape(title)}</figcaption>'
        f'<img src="data:image/png;base64,{data}" alt="{html.escape(title)}" /></figure>'
    )


def _top_table(df_top: pd.DataFrame, limit: int = 25) -> str:
    cols = ["source_id", "anomaly_score"] + [c for c in _ASTRO_COLS if c in df_top.columns]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body_rows = []
    for _, r in df_top.head(limit).iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if c == "source_id":
                cells.append(f"<td><code>{html.escape(str(int(v)))}</code></td>")
            else:
                try:
                    fv = float(v)
                    cells.append("<td>{}</td>".format("—" if not np.isfinite(fv) else f"{fv:.4g}"))
                except (TypeError, ValueError):
                    cells.append("<td>—</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _stat_cards(summary: Dict[str, Any]) -> str:
    c = summary.get("counts", {})
    cards = [
        ("Sources", c.get("n_rows", 0)),
        ("Anomalies", c.get("n_anomalies", 0)),
        ("Fraction", f"{100 * float(c.get('anomaly_fraction', 0.0)):.2f} %"),
        ("Nœuds", c.get("n_nodes", 0)),
        ("Arêtes", c.get("n_edges", 0)),
    ]
    return "".join(
        f'<div class="card"><div class="num">{html.escape(str(v))}</div>'
        f'<div class="lbl">{html.escape(label)}</div></div>'
        for label, v in cards
    )


def write_report(
    out_dir: str | Path,
    summary: Dict[str, Any],
    df_top: pd.DataFrame,
    plots_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Write summary.json + report.html. Returns the artefact filenames."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figures = ""
    if plots_dir is not None and plots_dir.is_dir():
        ordered = [n for n in _PLOT_TITLES if (plots_dir / n).exists()]
        extras = sorted(p.name for p in plots_dir.glob("*.png") if p.name not in _PLOT_TITLES)
        tags = [_img_tag(plots_dir / n, _PLOT_TITLES[n]) for n in ordered]
        tags += [_img_tag(plots_dir / n, n) for n in extras]
        if tags:
            figures = f'<section><h2>Visualisations</h2><div class="figs">{"".join(tags)}</div></section>'

    engine = html.escape(str(summary.get("engine", "")))
    emb = html.escape(str(summary.get("embedding_mode", "")))
    doc = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AstroGraphAnomaly — rapport de run</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; max-width: 1100px; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 14px; margin-top: 0; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }}
  .card {{ flex: 1 1 120px; background: #f6f8fa; border-radius: 12px; padding: 14px 16px; text-align: center; }}
  .card .num {{ font-size: 24px; font-weight: 700; }}
  .card .lbl {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .04em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e6e6e6; padding: 7px 9px; text-align: left; }}
  th {{ background: #fafafa; position: sticky; top: 0; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 13px; color: #555; margin-bottom: 6px; }}
  img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,.12); }}
  pre {{ background: #f6f8fa; padding: 14px; border-radius: 10px; overflow-x: auto; font-size: 12px; }}
  section {{ margin-top: 28px; }}
</style>
</head>
<body>
  <h1>AstroGraphAnomaly — rapport de run</h1>
  <p class="meta">Engine: <b>{engine}</b> · embedding: <b>{emb}</b> · généré: {html.escape(str(summary.get("generated_at", "")))}</p>
  <div class="cards">{_stat_cards(summary)}</div>

  <section>
    <h2>Top anomalies</h2>
    {_top_table(df_top)}
  </section>

  {figures}

  <section>
    <h2>Résumé (JSON)</h2>
    <pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  </section>
</body>
</html>
"""
    (out / "report.html").write_text(doc, encoding="utf-8")
    return {"summary": "summary.json", "report": "report.html"}
