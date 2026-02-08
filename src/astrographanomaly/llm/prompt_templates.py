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

def build_prompt(payload: Dict[str, Any], template_name: str = "default") -> str:
    template = DEFAULT_TEMPLATE
    lime_weights = payload.get("lime", {}).get("weights", [])
    lime_lines = "\n".join([f"- {w['feature']}: {w['weight']:+.4f}" for w in lime_weights]) if lime_weights else "- (none)"
    feat_snap = payload.get("feature_snapshot", {})
    feat_lines = "\n".join([f"- {k}: {v:+.4f}" for k, v in feat_snap.items()]) if feat_snap else "- (none)"

    return template.format(
        source_id=payload.get("source_id"),
        anomaly_score=payload.get("anomaly_score"),
        engine=payload.get("engine"),
        threshold_strategy=payload.get("threshold_strategy"),
        feature_snapshot=feat_lines,
        lime_weights=lime_lines,
    )
