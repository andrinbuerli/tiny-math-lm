"""Read-only helpers for browsing prepared data bundles."""

from __future__ import annotations

import json
from pathlib import Path


def discover_bundles(data_dir: Path) -> list[Path]:
    """Return prepared data directories that contain a readable manifest."""
    if not data_dir.exists():
        return []
    manifests = list(data_dir.glob("processed/*/manifest.json"))
    manifests.sort(
        key=lambda manifest: (manifest.stat().st_mtime_ns, manifest.parent.name), reverse=True
    )
    return [manifest.parent for manifest in manifests]


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Read a prepared split, ignoring blank lines and reporting malformed rows."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object on line {line_number} of {path}")
            rows.append({str(key): str(value) for key, value in row.items()})
    return rows
