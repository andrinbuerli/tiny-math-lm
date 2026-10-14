"""Atomic checkpoint persistence including optimizer and random state."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

COMPATIBILITY_FIELDS = (
    "protocol",
    "annotation_sha256",
    "tokenizer_sha256",
    "bundle_sha256",
    "context_length",
)


def compatibility_metadata(
    *,
    protocol: str,
    annotation_sha256: str | None,
    tokenizer_sha256: str,
    bundle_sha256: str,
    context_length: int,
) -> dict[str, Any]:
    """Build the immutable dataset/protocol identity stored in checkpoints."""
    if protocol not in {"answer_only", "reasoning_v1"}:
        raise ValueError(f"Unsupported checkpoint protocol: {protocol}")
    if not tokenizer_sha256 or not bundle_sha256 or context_length < 1:
        raise ValueError("Checkpoint compatibility metadata is incomplete")
    if protocol == "reasoning_v1" and not annotation_sha256:
        raise ValueError("Reasoning checkpoints require an annotation identity")
    return {
        "protocol": protocol,
        "annotation_sha256": annotation_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "bundle_sha256": bundle_sha256,
        "context_length": int(context_length),
    }


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any], expected: dict[str, Any], *, allow_legacy_answer_only: bool = True
) -> None:
    """Reject checkpoints whose protocol or data/model identity differs.

    Old answer-only checkpoints predate explicit compatibility metadata. They
    remain loadable when their tokenizer digest and model context agree.
    Reasoning checkpoints must carry every field because those fields define
    the annotation and leakage boundary.
    """
    if set(expected) != set(COMPATIBILITY_FIELDS):
        raise ValueError("Expected compatibility metadata must contain all identity fields")
    actual = checkpoint.get("compatibility")
    if actual is None:
        if allow_legacy_answer_only and expected["protocol"] == "answer_only":
            if checkpoint.get("tokenizer_sha256") != expected["tokenizer_sha256"]:
                raise ValueError("Checkpoint tokenizer does not match the evaluation data bundle")
            config_context = checkpoint.get("model_config", {}).get("context_length")
            if config_context is not None and int(config_context) != int(expected["context_length"]):
                raise ValueError("Checkpoint context length does not match the evaluation bundle")
            return
        actual = {field: checkpoint.get(field) for field in COMPATIBILITY_FIELDS}
    if not any(actual.values()):
        if allow_legacy_answer_only and expected["protocol"] == "answer_only":
            if checkpoint.get("tokenizer_sha256") != expected["tokenizer_sha256"]:
                raise ValueError("Checkpoint tokenizer does not match the evaluation data bundle")
            config_context = checkpoint.get("model_config", {}).get("context_length")
            if config_context is not None and int(config_context) != int(expected["context_length"]):
                raise ValueError("Checkpoint context length does not match the evaluation bundle")
            return
        raise ValueError("Checkpoint is missing required compatibility metadata")
    required_fields = (
        COMPATIBILITY_FIELDS
        if expected["protocol"] == "reasoning_v1"
        else tuple(field for field in COMPATIBILITY_FIELDS if field != "annotation_sha256")
    )
    missing = [field for field in required_fields if field not in actual or actual[field] is None]
    if missing:
        raise ValueError(f"Checkpoint compatibility metadata is missing: {', '.join(missing)}")
    mismatches = [field for field in COMPATIBILITY_FIELDS if actual[field] != expected[field]]
    if mismatches:
        field = mismatches[0]
        raise ValueError(
            f"Checkpoint {field} does not match evaluation data "
            f"(checkpoint={actual[field]!r}, evaluation={expected[field]!r})"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Atomically save a resumable checkpoint under the workspace."""
    destination = Path(path).resolve()
    if Path("/workspace") not in destination.parents:
        raise ValueError("Checkpoint path must be inside /workspace")
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
