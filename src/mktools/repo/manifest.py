"""Local manifest helpers for :mod:`mktools.repo`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = ".mkrepo.manifest.json"


def manifest_path(dest: Path) -> Path:
    """Return the manifest path for a download destination."""

    return dest / MANIFEST_NAME


def read_manifest(dest: Path) -> dict[str, Any] | None:
    """Read a local manifest if present."""

    path = manifest_path(dest)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(dest: Path, manifest: dict[str, Any]) -> Path:
    """Write a local manifest and return its path."""

    dest.mkdir(parents=True, exist_ok=True)
    path = manifest_path(dest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path
