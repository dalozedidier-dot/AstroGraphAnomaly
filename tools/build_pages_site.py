#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the GitHub Pages site from real AstroGraphAnomaly workflow outputs.

This script is intentionally small and CI-friendly:
- it reads the workflow YAML files under .github/workflows;
- it reads one generated run directory, usually results/<run_name>;
- it copies that run into the static site;
- it writes HTML pages that reflect the actual pipeline/workflow contract.

The page is not a hand-written model brochure. It is generated from the same
entrypoint and artefacts used by the GitHub Actions workflows.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - requirements.txt includes pyyaml
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]

ENGINE_NOTES: Dict[str, Dict[str, str]] = {
    "isolation_forest": {
        "label": "Isolation Forest",
        "role": "détection généraliste d’objets isolés dans l’espace de features graphe + astro",
        "when": "bon choix par défaut pour un run propre et lisible",
        "source": "src/astrographanomaly/detection/isoforest.py",
    },
    "lof": {
        "label": "LOF",
        "role": "cherche les points dont le voisinage local est atypique",
        "when": "utile quand l’anomalie est locale plutôt que globale",
        "source": "src/astrographanomaly/detection/lof.py",
    },
    "ocsvm": {
        "label": "One-Class SVM",
        "role": "sépare une région normale d’un extérieur statistique",
        "when": "intéressant comme lecture complémentaire, plus sensible aux paramètres",
        "source": "src/astrographanomaly/detection/ocsvm.py",
    },
    "robust_zscore": {
        "label": "Robust Z-Score",
        "role": "score robuste basé sur l’écart aux médianes et MAD",
        "when": "excellent moteur de contrôle, rapide, interprétable et stable en CI",
        "source": "src/astrographanomaly/detection/robust.py",
    },
    "pineforest": {
        "label": "PineForest",
        "role": "moteur optionnel via coniferest pour une forêt d’anomalies alternative",
        "when": "à utiliser seulement si la dépendance optionnelle coniferest est installée",
        "source": "src/astrographanomaly/detection/pineforest.py",
    },
    "ensemble": {
        "label": "Ensemble / incoherence score",
        "role": "fusionne plusieurs contraintes : moteurs statistiques + contrainte graphe",
        "when": "le plus cohérent pour croiser plusieurs signaux au lieu de croire un seul score",
        "source": "src/astrographanomaly/detection/ensemble.py",
    },
}

WORKFLOW_HINTS: Dict[str, str] = {
    "ci.yml": "Tests unitaires + smoke test du pipeline sur CSV exemple.",
    "ci_full_artifacts.yml": "Pipeline complet offline + artefacts de plots sur plusieurs versions Python.",
    "matrix.yml": "Matrice Python pour valider le contrat minimal du pipeline.",
    "notebook_smoke.yml": "Validation rapide des notebooks/chemins Colab.",
    "plots.yml": "Génération et vérification des plots offline.",
    "manual_real_data_showcase.yml": "Run manuel Gaia DR3 autour d’une zone RA/Dec, avec rapport et visualisations.",
    "manual_viz_a_to_h.yml": "Run manuel orienté visualisations A→H à partir du pipeline.",
    "real_gaia_cone_stars.yml": "Run réel Gaia DR3 sur un cône d’étoiles.",
    "real_galaxy_candidates_5p.yml": "Sélection réelle de candidats galaxies avec critères 5 paramètres.",
    "real_quasar_candidates_5p.yml": "Sélection réelle de candidats quasars avec critères 5 paramètres.",
    "real_ruwe_outliers.yml": "Recherche d’objets atypiques via RUWE/outliers Gaia.",
    "real_variability.yml": "Recherche d’objets variables Gaia via vari_summary.",
    "pages.yml": "Génère ce site depuis les sorties réelles du workflow Pages.",
}


def h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_cli_choices() -> Dict[str, List[str]]:
    text = read_text(REPO_ROOT / "run_workflow.py") + "\n" + read_text(REPO_ROOT / "src/astrographanomaly/cli.py")
    out: Dict[str, List[str]] = {}
    for arg_name in ("--engine", "--threshold-strategy", "--features-mode"):
        idx = text.find(f'"{arg_name}"')
        if idx < 0:
            idx = text.find(f"'{arg_name}'")
        block = text[idx : idx + 900] if idx >= 0 else ""
        m = re.search(r"choices=\[(.*?)\]", block, flags=re.S)
        if not m:
            continue
        out[arg_name.lstrip("-").replace("-", "_")] = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
    return out


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_top_rows(path: Path, limit: int = 12) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: v for k, v in row.items()})
                if len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return h(value)
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:.{digits}g}"


def file_size(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return ""
    units = ["B", "KB", "MB", "GB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return str(n)


def yaml_on_block(data: Mapping[str, Any]) -> Any:
    # PyYAML YAML 1.1 can parse the key `on:` as boolean True.
    return data.get("on", data.get(True, {}))


def normalize_trigger(on_block: Any) -> str:
    if isinstance(on_block, str):
        return on_block
    if isinstance(on_block, list):
        return ", ".join(map(str, on_block))
    if isinstance(on_block, dict):
        return ", ".join(str(k) for k in on_block.keys())
    return ""


def extract_inputs(on_block: Any) -> Dict[str, Any]:
    if not isinstance(on_block, dict):
        return {}
    wd = on_block.get("workflow_dispatch")
    if isinstance(wd, dict):
        inputs = wd.get("inputs")
        if isinstance(inputs, dict):
            return inputs
    return {}


def summarize_workflows() -> List[Dict[str, Any]]:
    workflows: List[Dict[str, Any]] = []
    wf_dir = REPO_ROOT / ".github" / "workflows"
    for path in sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml")):
        raw = read_text(path)
        parsed: Dict[str, Any] = {}
        if yaml is not None:
            try:
                loaded = yaml.safe_load(raw) or {}
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                parsed = {}
        on_block = yaml_on_block(parsed)
        inputs = extract_inputs(on_block)
        engine_default = ""
        if isinstance(inputs.get("engine"), dict):
            engine_default = str(inputs["engine"].get("default", ""))
        workflows.append(
            {
                "file": path.name,
                "name": parsed.get("name") or path.stem,
                "trigger": normalize_trigger(on_block),
                "engine_default": engine_default,
                "input_count": len(inputs),
                "has_artifacts": "upload-artifact" in raw,
                "has_pages": "deploy-pages" in raw or "upload-pages-artifact" in raw,
                "uses_gaia": "gaiadr3" in raw.lower() or "gaia" in path.name.lower(),
                "hint": WORKFLOW_HINTS.get(path.name, "Workflow GitHub Actions du dépôt."),
            }
        )
    return workflows


def pills_html(values: Iterable[Any]) -> str:
    return "".join(f'<span class="pill">{h(x)}</span>' for x in values)


def collect_artifacts(run_dir: Path, site_run_dir: str) -> List[Dict[str, str]]:
    important = [
        ("Rapport HTML", "report.html", "Rapport auto-généré, lisible directement dans le navigateur."),
        ("Résumé JSON", "summary.json", "Synthèse machine-readable du run."),
        ("Manifest", "manifest.json", "Contrat des artefacts produits par le pipeline."),
        ("Top anomalies", "top_anomalies.csv", "Candidats classés par score décroissant."),
        ("Scored CSV", "scored.csv", "Toutes les sources avec score et label d’anomalie."),
        ("Raw CSV", "raw.csv", "Données d’entrée normalisées par le pipeline."),
        ("Graphe complet", "graph_full.graphml", "Graphe kNN complet avec attributs d’anomalie."),
        ("Graphe top-k", "graph_topk.graphml", "Sous-graphe des meilleurs candidats."),
        ("Explications", "explanations.jsonl", "Explications locales si explain_top > 0."),
        ("Prompts LLM", "llm_prompts.jsonl", "Prompts générés pour revue/interprétation."),
    ]
    rows: List[Dict[str, str]] = []
    for label, rel, desc in important:
        path = run_dir / rel
        if path.exists():
            rows.append({"label": label, "href": f"{site_run_dir}/{rel}", "desc": desc, "size": file_size(path)})
    plots = run_dir / "plots"
    if plots.exists():
        for p in sorted(plots.glob("*.png"))[:12]:
            rows.append({"label": f"Plot: {p.stem}", "href": f"{site_run_dir}/plots/{p.name}", "desc": "Image produite par le workflow.", "size": file_size(p)})
    return rows


def copy_run(run_dir: Path, out_dir: Path, run_name: str) -> str:
    site_run_rel = f"runs/{run_name}"
    dest = out_dir / site_run_rel
    if dest.exists():
        shutil.rmtree(dest)
    if run_dir.exists():
        shutil.copytree(run_dir, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)
    return site_run_rel


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def base_css() -> str:
    return """
:root{--bg:#050716;--panel:#0d1329;--panel2:#111936;--line:#263253;--text:#eef3ff;--muted:#aab6d8;--accent:#6ee7ff;--accent2:#c084fc;--warn:#ffd166;--ok:#7ee787}*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:radial-gradient(circle at 20% 0%,#18204a 0,#050716 38%),linear-gradient(160deg,#050716,#091126);color:var(--text);line-height:1.55}.wrap{width:min(1180px,92vw);margin:auto}.hero{padding:72px 0 36px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.14em;text-transform:uppercase;font-size:.8rem}.hero h1{font-size:clamp(2rem,5vw,4.8rem);line-height:.95;margin:.25em 0}.lead{font-size:clamp(1.05rem,2vw,1.35rem);max-width:850px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}.card{grid-column:span 4;background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.025));border:1px solid var(--line);border-radius:22px;padding:20px;box-shadow:0 16px 40px rgba(0,0,0,.25)}.wide{grid-column:span 8}.full{grid-column:1/-1}.card h2,.card h3{margin-top:0}.muted{color:var(--muted)}.pill{display:inline-flex;gap:6px;align-items:center;padding:5px 10px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.04);color:var(--muted);font-size:.88rem;margin:2px 4px 2px 0}.stat{font-size:2rem;font-weight:800;color:var(--accent)}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:16px}table{width:100%;border-collapse:collapse;min-width:760px}th,td{text-align:left;padding:11px 13px;border-bottom:1px solid rgba(255,255,255,.08);vertical-align:top}th{color:#dce7ff;background:rgba(255,255,255,.06)}td{color:var(--muted)}code,pre{background:#080d1d;border:1px solid var(--line);border-radius:10px;color:#dbeafe}code{padding:2px 6px}pre{padding:14px;overflow:auto}.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.arrow{color:var(--accent2);font-weight:900}.footer{padding:48px 0;color:var(--muted)}.artifact-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.artifact{border:1px solid var(--line);background:rgba(255,255,255,.04);border-radius:16px;padding:14px}.artifact strong{display:block;color:var(--text)}.plot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.plot-grid img{width:100%;height:auto;border:1px solid var(--line);border-radius:16px;background:#fff}.small{font-size:.9rem}@media(max-width:900px){.card,.wide{grid-column:1/-1}.hero{padding-top:42px}}
""".strip()


def layout(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang=\"fr\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{h(title)}</title>
  <style>{base_css()}</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_index(
    *,
    run_name: str,
    site_run_rel: str,
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
    workflows: List[Dict[str, Any]],
    engines: List[str],
    thresholds: List[str],
    artifacts: List[Dict[str, str]],
    top_rows: List[Dict[str, str]],
    repo: str,
    run_url: str,
) -> str:
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    counts = summary.get("counts") or {}
    score_stats = summary.get("score_stats") or {}
    engine = summary.get("engine") or (manifest.get("config") or {}).get("engine", "")
    threshold = summary.get("threshold") or (manifest.get("config") or {}).get("threshold", {})

    workflow_rows = []
    for wf in workflows:
        link = f"https://github.com/{repo}/actions/workflows/{wf['file']}" if repo else f".github/workflows/{wf['file']}"
        flags = []
        if wf.get("has_artifacts"):
            flags.append("artefacts")
        if wf.get("has_pages"):
            flags.append("pages")
        if wf.get("uses_gaia"):
            flags.append("Gaia")
        if wf.get("engine_default"):
            flags.append(f"engine: {wf['engine_default']}")
        workflow_rows.append(
            f"<tr><td><a href=\"{h(link)}\">{h(wf['name'])}</a><br><span class=\"small muted\">{h(wf['file'])}</span></td>"
            f"<td>{h(wf['trigger'])}</td><td>{h(wf['hint'])}</td><td>{pills_html(flags)}</td></tr>"
        )

    engine_cards = []
    for eng in engines:
        note = ENGINE_NOTES.get(eng, {"label": eng, "role": "Moteur disponible dans le CLI.", "when": "", "source": ""})
        engine_cards.append(
            f"<div class=\"card\"><h3>{h(note['label'])}</h3>"
            f"<p class=\"muted\">{h(note['role'])}</p>"
            f"<p><span class=\"pill\">{h(eng)}</span></p>"
            f"<p class=\"small\"><strong>Usage :</strong> {h(note.get('when',''))}</p>"
            f"<p class=\"small muted\"><code>{h(note.get('source',''))}</code></p></div>"
        )

    artifact_html = "".join(
        f"<div class=\"artifact\"><strong><a href=\"{h(a['href'])}\">{h(a['label'])}</a></strong>"
        f"<span class=\"muted small\">{h(a['desc'])}</span><br><span class=\"pill\">{h(a['size'])}</span></div>"
        for a in artifacts
    ) or "<p class=\"muted\">Aucun artefact détecté. Le workflow Pages doit d’abord exécuter le pipeline.</p>"

    plot_imgs = []
    for a in artifacts:
        if a["href"].lower().endswith(".png"):
            plot_imgs.append(f"<a href=\"{h(a['href'])}\"><img src=\"{h(a['href'])}\" alt=\"{h(a['label'])}\"></a>")
    plot_grid = "".join(plot_imgs[:6]) or "<p class=\"muted\">Les plots apparaîtront ici après un run avec <code>--plots</code>.</p>"

    top_table = ""
    if top_rows:
        cols = [c for c in ["source_id", "anomaly_score", "incoherence_score", "anomaly_label", "ra", "dec", "parallax", "phot_g_mean_mag", "bp_rp", "ruwe"] if c in top_rows[0]]
        head = "".join(f"<th>{h(c)}</th>" for c in cols)
        rows = []
        for row in top_rows:
            rows.append("<tr>" + "".join(f"<td>{fmt_float(row.get(c,''))}</td>" for c in cols) + "</tr>")
        top_table = f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    else:
        top_table = "<p class=\"muted\">Aucun top_anomalies.csv détecté.</p>"

    run_link = f"<a href=\"{h(run_url)}\">run GitHub Actions</a>" if run_url else "run GitHub Actions"
    report_link = f"{site_run_rel}/report.html"

    body = f"""
<div class=\"wrap\">
  <section class=\"hero\">
    <div class=\"eyebrow\">AstroGraphAnomaly · workflow-first</div>
    <h1>Page générée par les workflows, pas une vitrine séparée.</h1>
    <p class=\"lead\">Ce site est construit depuis les fichiers <code>.github/workflows</code> et depuis un vrai dossier <code>results/{h(run_name)}</code> produit par <code>run_workflow.py</code>. Il expose donc le contrat réel : données → graphe kNN → features → moteur → seuil → artefacts.</p>
    <p class=\"muted small\">Généré le {h(generated)} · {run_link}</p>
  </section>

  <section class=\"grid\">
    <article class=\"card full\">
      <h2>Chaîne publiée</h2>
      <div class=\"flow\">
        <span class=\"pill\">GitHub Actions</span><span class=\"arrow\">→</span>
        <span class=\"pill\">run_workflow.py</span><span class=\"arrow\">→</span>
        <span class=\"pill\">results/{h(run_name)}</span><span class=\"arrow\">→</span>
        <span class=\"pill\">tools/build_pages_site.py</span><span class=\"arrow\">→</span>
        <span class=\"pill\">GitHub Pages</span>
      </div>
    </article>

    <article class=\"card\"><h3>Lignes analysées</h3><div class=\"stat\">{h(counts.get('n_rows','—'))}</div><p class=\"muted\">sources dans le run publié</p></article>
    <article class=\"card\"><h3>Anomalies top-k</h3><div class=\"stat\">{h(counts.get('n_anomalies','—'))}</div><p class=\"muted\">candidats exportés</p></article>
    <article class=\"card\"><h3>Graphe</h3><div class=\"stat\">{h(counts.get('n_edges','—'))}</div><p class=\"muted\">arêtes kNN</p></article>

    <article class=\"card wide\">
      <h2>Dernier run publié</h2>
      <p><span class=\"pill\">engine: {h(engine)}</span><span class=\"pill\">threshold: {h(threshold.get('strategy',''))}</span><span class=\"pill\">features: {h(summary.get('features_mode',''))}</span><span class=\"pill\">seed: {h(summary.get('seed',''))}</span></p>
      <p class=\"muted\">Score min/p50/p99/max : <code>{fmt_float(score_stats.get('min'))}</code> · <code>{fmt_float(score_stats.get('p50'))}</code> · <code>{fmt_float(score_stats.get('p99'))}</code> · <code>{fmt_float(score_stats.get('max'))}</code></p>
      <p><a href=\"{h(report_link)}\">Ouvrir le rapport HTML généré</a></p>
    </article>
    <article class=\"card\">
      <h2>Seuils disponibles</h2>
      <p>{pills_html(thresholds)}</p>
      <p class=\"muted small\">Le seuil transforme un score en liste exploitable. Ce n’est pas une preuve physique, c’est un triage.</p>
    </article>

    <article class=\"card full\">
      <h2>Artefacts réellement produits</h2>
      <div class=\"artifact-list\">{artifact_html}</div>
    </article>

    <article class=\"card full\">
      <h2>Top anomalies du run publié</h2>
      {top_table}
    </article>

    <article class=\"card full\">
      <h2>Plots générés par le workflow</h2>
      <div class=\"plot-grid\">{plot_grid}</div>
    </article>

    <article class=\"card full\">
      <h2>Moteurs réellement exposés par le pipeline</h2>
      <p class=\"muted\">Liste extraite de <code>run_workflow.py</code> / CLI, pas recopiée à la main dans une page déconnectée.</p>
    </article>
    {''.join(engine_cards)}

    <article class=\"card full\">
      <h2>Workflows du dépôt</h2>
      <div class=\"table-wrap\"><table><thead><tr><th>Workflow</th><th>Déclencheurs</th><th>Rôle</th><th>Signaux</th></tr></thead><tbody>{''.join(workflow_rows)}</tbody></table></div>
    </article>

    <article class=\"card full\">
      <h2>Commande équivalente du run Pages</h2>
      <pre><code>python run_workflow.py --mode csv \\
  --in-csv data/sample_gaia_like.csv \\
  --out results/{h(run_name)} \\
  --engine {h(engine or 'ensemble')} \\
  --threshold-strategy {h(threshold.get('strategy','top_k'))} \\
  --top-k {h(threshold.get('top_k','20'))} \\
  --knn-k 8 --features-mode extended --plots --explain-top 5</code></pre>
    </article>
  </section>

  <footer class=\"footer\">
    <p>AstroGraphAnomaly publie des signaux d’anomalie. Un score élevé n’est pas une preuve astrophysique ; c’est une priorité de revue.</p>
  </footer>
</div>
"""
    return layout("AstroGraphAnomaly · Workflow Pages", body)


def build_workflow_page(workflows: List[Dict[str, Any]], repo: str) -> str:
    rows = []
    for wf in workflows:
        link = f"https://github.com/{repo}/actions/workflows/{wf['file']}" if repo else f".github/workflows/{wf['file']}"
        rows.append(
            f"<tr><td><a href=\"{h(link)}\">{h(wf['name'])}</a><br><span class=\"small muted\">{h(wf['file'])}</span></td>"
            f"<td>{h(wf['trigger'])}</td><td>{h(wf['input_count'])}</td><td>{h(wf['hint'])}</td></tr>"
        )
    body = f"""
<div class=\"wrap\"><section class=\"hero\"><div class=\"eyebrow\">AstroGraphAnomaly</div><h1>Workflows</h1><p class=\"lead\">Inventaire généré depuis <code>.github/workflows</code>.</p><p><a href=\"index.html\">← Retour dashboard</a></p></section><div class=\"card full\"><div class=\"table-wrap\"><table><thead><tr><th>Workflow</th><th>Déclencheurs</th><th>Inputs</th><th>Rôle</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div></div>
"""
    return layout("AstroGraphAnomaly · Workflows", body)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/pages_demo")
    ap.add_argument("--out", default="_site")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    run_name = args.run_name or run_dir.name or "pages_demo"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    site_run_rel = copy_run(run_dir, out_dir, run_name)
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "manifest.json")
    top_rows = load_top_rows(run_dir / "top_anomalies.csv")
    choices = parse_cli_choices()
    workflows = summarize_workflows()
    artifacts = collect_artifacts(run_dir, site_run_rel)

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_url = ""
    if repo and os.environ.get("GITHUB_RUN_ID"):
        run_url = f"https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"

    write(
        out_dir / "index.html",
        build_index(
            run_name=run_name,
            site_run_rel=site_run_rel,
            summary=summary,
            manifest=manifest,
            workflows=workflows,
            engines=choices.get("engine", list(ENGINE_NOTES.keys())),
            thresholds=choices.get("threshold_strategy", ["contamination", "percentile", "top_k", "score"]),
            artifacts=artifacts,
            top_rows=top_rows,
            repo=repo,
            run_url=run_url,
        ),
    )
    write(out_dir / "workflows.html", build_workflow_page(workflows, repo))

    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": run_name,
        "run_dir": str(run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir),
        "engines": choices.get("engine", list(ENGINE_NOTES.keys())),
        "workflows": workflows,
    }
    write(out_dir / "site_manifest.json", json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[build_pages_site] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
