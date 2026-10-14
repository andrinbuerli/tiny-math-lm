"""Helpers for checkpoint-aligned local metric histories."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


def elapsed_seconds_through_step(path: Path, *, step_column: str, step: int) -> float:
    """Return cumulative elapsed time through a checkpoint's metric row, if present."""
    if not path.is_file():
        return 0.0
    elapsed = 0.0
    matched = False
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or step_column not in reader.fieldnames:
            raise ValueError(f"Metric CSV {path} has no {step_column!r} step column")
        for row in reader:
            raw_step = row.get(step_column)
            if raw_step is None or not raw_step.strip():
                continue
            row_step = int(raw_step)
            if row_step <= step:
                raw_elapsed = row.get("wall_seconds")
                if raw_elapsed is not None and raw_elapsed.strip():
                    elapsed = float(raw_elapsed)
                    matched = True
    return elapsed if matched else 0.0


def truncate_csv_after_step(path: Path, *, step_column: str, step: int) -> None:
    """Drop metrics newer than a resumed checkpoint before replaying optimizer steps."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or step_column not in fieldnames:
            raise ValueError(f"Metric CSV {path} has no {step_column!r} step column")
        rows_by_step: dict[int, dict[str, str]] = {}
        for row in reader:
            raw_step = row.get(step_column)
            if raw_step is None or not raw_step.strip():
                continue
            row_step = int(raw_step)
            if row_step <= step:
                # Keep the original row if a prior resume already replayed it.
                rows_by_step.setdefault(row_step, row)
        rows = [rows_by_step[key] for key in sorted(rows_by_step)]

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
