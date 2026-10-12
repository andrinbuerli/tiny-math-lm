"""Checkpoint persistence and experiment metadata."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinymath.repro import runtime_info


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically save a resumable checkpoint within the caller's workspace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def write_run_metadata(path: str | Path, config: dict[str, Any]) -> None:
    """Write effective config and observed runtime versions as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"config": config, "runtime": runtime_info().as_dict()}
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
