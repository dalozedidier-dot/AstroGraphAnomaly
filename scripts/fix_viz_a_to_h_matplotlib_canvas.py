#!/usr/bin/env python3
"""Patch AstroGraphAnomaly tools/viz_a_to_h_suite.py for Matplotlib FigureCanvasAgg API change.

Fixes:
- Replace fig.canvas.tostring_rgb() usage with fig.canvas.buffer_rgba() + reshape.

Why:
Recent Matplotlib versions removed FigureCanvasAgg.tostring_rgb().
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def patch_file(path: pathlib.Path) -> int:
    text = path.read_text(encoding="utf-8")

    if "tostring_rgb" not in text:
        print("[INFO] No 'tostring_rgb' found. Nothing to patch.")
        return 0

    # Find the exact frombuffer(tostring_rgb()) line and replace it with a robust RGBA buffer capture.
    lines = text.splitlines(True)  # keep line endings

    pat = re.compile(r"^(?P<indent>\s*)img\s*=\s*np\.frombuffer\(\s*fig\.canvas\.tostring_rgb\(\)\s*,\s*dtype\s*=\s*np\.uint8\s*\)\s*$")

    out: list[str] = []
    replaced = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        m = pat.match(line.rstrip("\n\r"))
        if m:
            indent = m.group("indent")
            eol = "\n" if line.endswith("\n") else ""
            block = [
                f"{indent}fig.canvas.draw(){eol}",
                f"{indent}w, h = fig.canvas.get_width_height(){eol}",
                f"{indent}{eol}",
                f"{indent}buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8){eol}",
                f"{indent}img = buf.reshape((h, w, 4))[:, :, :3].copy(){eol}",
            ]
            out.extend(block)
            replaced += 1

            # Skip any immediate reshape lines that were used for tostring_rgb.
            j = i + 1
            skipped = 0
            while j < len(lines) and skipped < 3:
                nxt = lines[j].strip()
                if (
                    nxt.startswith("img = img.reshape")
                    or "get_width_height" in nxt and "reshape" in nxt and "img" in nxt
                    or "tostring_rgb" in nxt
                ):
                    j += 1
                    skipped += 1
                    continue
                break
            i = j
            continue

        out.append(line)
        i += 1

    if replaced == 0:
        print("[WARN] 'tostring_rgb' exists, but expected pattern not found. No changes applied.")
        return 2

    new_text = "".join(out)

    # Safety: ensure we removed tostring_rgb usage.
    if "tostring_rgb" in new_text:
        print("[WARN] Patch applied, but 'tostring_rgb' still present somewhere. You may have multiple occurrences.")

    # Write back with a .bak.
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(text, encoding="utf-8")
    path.write_text(new_text, encoding="utf-8")

    print(f"[OK] Patched {path} (backup: {bak.name}). Replacements: {replaced}")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--file",
        default="tools/viz_a_to_h_suite.py",
        help="Path to viz_a_to_h_suite.py (default: tools/viz_a_to_h_suite.py)",
    )
    args = p.parse_args(argv)

    path = pathlib.Path(args.file)
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return 1

    return patch_file(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
