"""Helpers for linking local training artifacts to their W&B runs."""

from __future__ import annotations

import json
from pathlib import Path


def read_wandb_run_id(run_dir: str | Path) -> str | None:
    """Read a persisted W&B run ID from a training or evaluation directory."""
    state_path = Path(run_dir) / "wandb_run.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_id = state.get("id") if isinstance(state, dict) else None
    return run_id if isinstance(run_id, str) and run_id else None
