"""Hydra configuration helpers shared by every experiment entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

WORKSPACE = Path("/workspace")


def workspace_path(*parts: str) -> Path:
    """Resolve a project path under the fixed experiment workspace."""
    path = WORKSPACE.joinpath(*parts).resolve()
    if path != WORKSPACE and WORKSPACE not in path.parents:
        raise ValueError(f"Path must stay under {WORKSPACE}")
    return path


def _sanitize(value: Any, parent_key: str = "") -> Any:
    """Remove credential-like fields recursively before persistence or logging."""
    sensitive = ("api_key", "apikey", "secret", "password", "credential", "token")
    if any(part in parent_key.lower() for part in sensitive):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(key): _sanitize(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, parent_key) for item in value]
    return value


def sanitized_config(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Return a resolved, credential-scrubbed plain mapping."""
    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    if not isinstance(raw, dict):
        raise TypeError("Experiment configuration must resolve to a mapping")
    return _sanitize(raw)


def persist_resolved_config(config: DictConfig, run_dir: str | Path) -> Path:
    """Write full sanitized composed config and return its path."""
    destination = Path(run_dir).resolve()
    if destination != WORKSPACE and WORKSPACE not in destination.parents:
        raise ValueError(f"Run directory must stay under {WORKSPACE}")
    destination.mkdir(parents=True, exist_ok=True)
    values = sanitized_config(config)
    path = destination / "resolved_config.json"
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def configure_workspace_environment() -> None:
    """Keep subprocesses and Hydra runs rooted at the project workspace."""
    os.environ["WORKSPACE_ROOT"] = str(WORKSPACE)
