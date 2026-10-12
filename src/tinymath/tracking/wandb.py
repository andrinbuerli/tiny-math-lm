"""Secure, resumable W&B run setup used by all experiment entry points."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values, load_dotenv
from omegaconf import DictConfig

from tinymath.config import persist_resolved_config, read_saved_resolved_config, sanitized_config
from tinymath.repro import runtime_info


class TrackingError(RuntimeError):
    """Raised when required experiment tracking cannot be initialized."""


_ACTIVE_STATE_PATHS: dict[str, Path] = {}


def _load_workspace_credentials() -> None:
    # python-dotenv parses standard assignment syntax and never prints the values.
    env_path = "/workspace/.env"
    load_dotenv(dotenv_path=env_path, override=False, verbose=False)
    if os.environ.get("WANDB_API_KEY"):
        return

    # Accept the legacy spelling present in this workspace's supplied .env.
    # Keep the value in process memory only; it is never added to config or metadata.
    legacy_key = dotenv_values(dotenv_path=env_path).get("WANDBKEY")
    if legacy_key:
        os.environ["WANDB_API_KEY"] = legacy_key


def _tracking_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingError(f"Cannot read W&B resume metadata at {path}: {exc}") from exc
    if not isinstance(state, dict) or not isinstance(state.get("id"), str):
        raise TrackingError(f"Invalid W&B resume metadata at {path}")
    return state


def _compatible_with_runtime_extension(
    saved: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Allow adding the runtime thread cap to runs created before that config group."""
    if "runtime" in saved or "runtime" not in current:
        return False
    without_new_group = dict(current)
    without_new_group.pop("runtime")
    return saved == without_new_group


def initialize_wandb(config: DictConfig, run_dir: str | Path, *, job_type: str) -> Any:
    """Create or resume one W&B run, persisting its ID before initialization."""
    import wandb

    _load_workspace_credentials()
    tracking = config.tracking
    mode = str(tracking.get("mode", "online"))
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    state_path = run_path / "wandb_run.json"
    state = _tracking_state(state_path)
    saved_config = read_saved_resolved_config(run_path)
    current_config = sanitized_config(config)
    if (
        state
        and saved_config is not None
        and saved_config != current_config
        and not _compatible_with_runtime_extension(saved_config, current_config)
    ):
        raise TrackingError(
            "This W&B run already has a different resolved Hydra config; resume it with the "
            "original options or choose a new run directory."
        )
    config_path = persist_resolved_config(config, run_path)
    run_id = state.get("id") or secrets.token_hex(8)
    if mode != "offline" and not os.environ.get("WANDB_API_KEY"):
        raise TrackingError(
            "W&B online tracking requires WANDB_API_KEY in the environment or /workspace/.env; "
            "no online run was started. Set tracking.mode=offline explicitly to create an "
            "unsynced local run."
        )
    state_path.write_text(
        json.dumps(
            {
                "id": run_id,
                "mode": mode,
                "sync_status": "unsynced" if mode == "offline" else "pending",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "runtime": runtime_info().as_dict(),
        "resolved_config_path": str(config_path),
        "protocol": "tinymath-r1",
        "resolved_config": sanitized_config(config),
    }
    try:
        run = wandb.init(
            project=str(tracking.get("project", "tinymath-rl")),
            entity=tracking.get("entity"),
            group=str(tracking.get("group", "default")),
            name=str(tracking.get("name", run_id)),
            tags=list(tracking.get("tags", [])),
            mode=mode,
            id=run_id,
            resume="allow" if tracking.get("resume", True) or state else None,
            dir=str(run_path),
            job_type=job_type,
            config=metadata,
            settings=wandb.Settings(code_dir="/workspace"),
        )
        _ACTIVE_STATE_PATHS[str(getattr(run, "id", run_id))] = state_path
        return run
    except (wandb.errors.Error, requests.exceptions.RequestException, OSError, ValueError) as exc:
        message = str(exc).replace(os.environ.get("WANDB_API_KEY", "\0"), "<redacted>")
        raise TrackingError(f"W&B initialization failed ({mode} mode): {message}") from None


def finish_wandb(run: Any) -> None:
    """Flush a run and report explicit offline state in local metadata."""
    if run is None:
        return
    mode = getattr(getattr(run, "settings", None), "mode", None)
    status = "unsynced" if str(mode) == "offline" else "synced"
    try:
        if hasattr(run, "summary"):
            run.summary["tracking/sync_status"] = status
        run.finish()
    finally:
        state_path = _ACTIVE_STATE_PATHS.pop(str(getattr(run, "id", "")), None)
        if state_path is not None and state_path.exists():
            state = _tracking_state(state_path)
            state["sync_status"] = status
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
