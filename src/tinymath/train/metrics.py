"""Helpers for checkpoint-aligned local metric histories."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


def truncate_csv_after_step(path: Path, *, step_column: str, step: int) -> None:
    """Drop metrics newer than a resumed checkpoint before replaying optimizer steps."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or step_column not in fieldnames:
            raise ValueError(f"Metric CSV {path} has no {step_column!r} step column")
        rows = [
            row
            for row in reader
            if row.get(step_column) and int(row[step_column]) <= step
        ]

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
