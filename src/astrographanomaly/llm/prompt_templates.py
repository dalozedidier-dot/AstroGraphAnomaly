from __future__ import annotations
from typing import Dict, Any, List

DEFAULT_TEMPLATE = """You are given a detected astronomical anomaly from a graph-based pipeline.
Summarize, in neutral technical language, why this source is anomalous based on the provided signals.
Do not assume physical causality; only restate the signals and their direction.

CONTEXT:
- source_id: {source_id}
- anomaly_score (higher = more anomalous): {anomaly_score}
- engine: {engine}
- threshold_strategy: {threshold_strategy}

FEATURE SNAPSHOT (scaled):
{feature_snapshot}

LIME (local linear surrogate) top weights:
{lime_weights}

OUTPUT FORMAT (JSON):
{{
  "source_id": {source_id},
  "summary": "...",
  "key_signals": [{{"signal": "...", "direction": "...", "strength": "low|med|high"}}],
  "caveats": ["..."]
}}
"""

COMPOSITE_TEMPLATE = """You are given a detected astronomical anomaly from a graph-based pipeline.
Summarize, in neutral technical language, why this source is anomalous based on the provided signals.
Do not assume physical causality; only restate the signals and their direction.

This run used a composite score ("incoherence_score"). Each constraint contributes a normalized violation in [0, 1] (phi), combined with a weight.

CONTEXT:
- source_id: {source_id}
- incoherence_score (0..1, higher = more anomalous): {incoherence_score}
- engine: {engine}
- threshold_strategy: {threshold_strategy}

CONSTRAINT BREAKDOWN:
{constraints}

FEATURE SNAPSHOT (scaled):
{feature_snapshot}

LIME (local linear surrogate) top weights:
{lime_weights}

OUTPUT FORMAT (JSON):
{{
  "source_id": {source_id},
  "summary": "...",
  "constraint_notes": [{{"constraint": "...", "note": "..."}}],
  "key_signals": [{{"signal": "...", "direction": "...", "strength": "low|med|high"}}],
  "caveats": ["..."]
}}
"""

def build_prompt(payload: Dict[str, Any], template_name: str = "default") -> str:
    template = DEFAULT_TEMPLATE if template_name != "composite" else COMPOSITE_TEMPLATE
    lime_weights = payload.get("lime", {}).get("weights", [])
    lime_lines = "\n".join([f"- {w['feature']}: {w['weight']:+.4f}" for w in lime_weights]) if lime_weights else "- (none)"
    feat_snap = payload.get("feature_snapshot", {})
    feat_lines = "\n".join([f"- {k}: {v:+.4f}" for k, v in feat_snap.items()]) if feat_snap else "- (none)"

    constraints_txt = "- (none)"
    if template_name == "composite":
        constraints = payload.get("constraints") or {}
        if isinstance(constraints, dict) and constraints:
            lines: List[str] = []
            # Sort by contribution weight*phi (descending) when possible.
            scored: List[tuple[str, float]] = []
            for k, v in constraints.items():
                if not isinstance(v, dict):
                    continue
                w = float(v.get("weight", 1.0))
                phi = float(v.get("phi", 0.0))
                scored.append((str(k), abs(w * phi)))
            order = [k for k, _ in sorted(scored, key=lambda t: t[1], reverse=True)]
            for k in order:
                v = constraints.get(k, {})
                if not isinstance(v, dict):
                    continue
                w = float(v.get("weight", 1.0))
                phi = float(v.get("phi", 0.0))
                raw = v.get("raw_score")
                try:
                    raw_f = float(raw)
                    raw_txt = f"{raw_f:.6g}"
                except Exception:
                    raw_txt = str(raw)
                lines.append(f"- {k}: weight={w:.3g}, phi={phi:.3f}, raw={raw_txt}")
            constraints_txt = "\n".join(lines) if lines else "- (none)"

    return template.format(
        source_id=payload.get("source_id"),
        anomaly_score=payload.get("anomaly_score"),
        incoherence_score=payload.get("incoherence_score", payload.get("anomaly_score")),
        engine=payload.get("engine"),
        threshold_strategy=payload.get("threshold_strategy"),
        feature_snapshot=feat_lines,
        lime_weights=lime_lines,
        constraints=constraints_txt,
    )
