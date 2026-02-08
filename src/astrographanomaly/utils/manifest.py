from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Dict, Any

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_manifest(out_dir: str, config: Dict[str, Any], artefacts: Dict[str, str]) -> None:
    out = Path(out_dir)
    checksums = {}
    for _, rel in artefacts.items():
        p = out / rel
        if p.exists() and p.is_file():
            checksums[rel] = sha256_file(p)

    manifest = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "config": config,
        "artefacts": artefacts,
        "checksums_sha256": checksums,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
