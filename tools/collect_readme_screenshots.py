#!/usr/bin/env python3
"""AstroGraphAnomaly: collect plots for README and generate a small HTML gallery.

Workflow-first utility:
- Works after a run exists in results/<run>/...
- Prefers results/<run>/viz_a_to_h/ if present, else results/<run>/plots/
- Copies a deterministic shortlist of "headline" images into screenshots/
- Generates screenshots/index.html

Usage:
  python tools/collect_readme_screenshots.py --run-dir results/<run>
  python tools/collect_readme_screenshots.py --run-dir results/<run> --out screenshots
  python tools/collect_readme_screenshots.py --run-dir results/<run> --list

Notes:
- No seaborn used.
- Does not modify your run directory; it only copies files out.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_HEADLINE_FILES = [
    # From classic plots suite
    "graph_communities_anomalies.png",
    "ra_dec_score.png",
    "pca_2d.png",
    "top_anomalies_scores.png",
    # From A->H suite (if present)
    "01_hidden_constellations_sky.png",
    "09_umap_cosmic_cloud.png",
]


def _find_source_dir(run_dir: Path) -> Tuple[Path, str]:
    viz_dir = run_dir / "viz_a_to_h"
    if viz_dir.is_dir():
        return viz_dir, "viz_a_to_h"
    plots_dir = run_dir / "plots"
    if plots_dir.is_dir():
        return plots_dir, "plots"
    raise FileNotFoundError(f"No plots directory found. Expected '{viz_dir}' or '{plots_dir}'.")


def _list_pngs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.png") if p.is_file()])


def _copy_if_exists(src_dir: Path, filename: str, dst_dir: Path) -> Optional[Path]:
    src = src_dir / filename
    if not src.exists():
        return None
    dst = dst_dir / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _write_index_html(dst_dir: Path, copied: List[Path], title: str) -> Path:
    index_path = dst_dir / "index.html"
    rel_items = [p.name for p in copied]
    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'/>",
        f"  <title>{title}</title>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "  <style>",
        "    body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:24px;}",
        "    h1{font-size:20px; margin:0 0 12px 0;}",
        "    .grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px;}",
        "    .card{border:1px solid #ddd; border-radius:12px; padding:12px; box-shadow: 0 1px 4px rgba(0,0,0,.06);}",
        "    img{width:100%; height:auto; border-radius:10px;}",
        "    .cap{font-size:12px; color:#444; margin-top:8px; word-break:break-all;}",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{title}</h1>",
        "  <div class='grid'>",
    ]
    for name in rel_items:
        parts += [
            "    <div class='card'>",
            f"      <img src='{name}' alt='{name}'/>",
            f"      <div class='cap'>{name}</div>",
            "    </div>",
        ]
    parts += ["  </div>", "</body>", "</html>"]
    index_path.write_text("\n".join(parts), encoding="utf-8")
    return index_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to results/<run>")
    ap.add_argument("--out", default="screenshots", help="Output directory for README screenshots (default: screenshots)")
    ap.add_argument("--headline-files", nargs="*", default=None, help="Override the default shortlist of headline PNGs to copy")
    ap.add_argument("--list", action="store_true", help="List available PNG files and exit")

    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out).resolve()

    src_dir, mode = _find_source_dir(run_dir)
    pngs = _list_pngs(src_dir)

    if args.list:
        print(f"Source directory: {src_dir} (mode={mode})")
        print(f"Found {len(pngs)} PNG files:")
        for p in pngs:
            print(" -", p.name)
        return 0

    headline_files = args.headline_files or DEFAULT_HEADLINE_FILES
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for fn in headline_files:
        dst = _copy_if_exists(src_dir, fn, out_dir)
        if dst is not None:
            copied.append(dst)

    if not copied:
        raise FileNotFoundError(
            "No headline PNGs were copied. Either the expected files are missing "
            "or your --headline-files list does not match your outputs."
        )

    title = f"AstroGraphAnomaly screenshots ({mode})"
    index = _write_index_html(out_dir, copied, title=title)

    print("Copied files:")
    for p in copied:
        print(" -", p)
    print("Gallery index:", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
