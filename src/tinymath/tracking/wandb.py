"""Secure, resumable W&B run setup used by all experiment entry points."""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values, load_dotenv
from omegaconf import DictConfig

from tinymath.config import persist_resolved_config, read_saved_resolved_config, sanitized_config
from tinymath.repro import runtime_info, seed_everything


class TrackingError(RuntimeError):
    """Raised when required experiment tracking cannot be initialized."""


LOGGER = logging.getLogger(__name__)


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


def _configure_wandb_service_threads(config: DictConfig) -> None:
    """Bound W&B's Go service thread pool for containers with tight PID limits."""
    runtime = config.get("runtime", {})
    max_procs = max(1, int(runtime.get("wandb_service_gomaxprocs", 2)))
    # Respect an explicit runtime environment override. GOMAXPROCS is read by
    # wandb-core when the SDK starts its Go service subprocess.
    os.environ.setdefault("GOMAXPROCS", str(max_procs))


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


def _write_tracking_state(path: Path, state: dict[str, Any]) -> None:
    """Replace local resume metadata atomically so a failed write preserves the run ID."""
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _compatible_with_runtime_extension(
    saved: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Accept config-only additions that cannot change a resumed training run."""
    import copy

    normalized = copy.deepcopy(current)
    saved_normalized = copy.deepcopy(saved)

    # This group was added to cap thread usage on resume. It does not change the
    # model, batches, optimizer, or objective.
    if "runtime" not in saved_normalized and "runtime" in normalized:
        normalized.pop("runtime")
    elif "runtime" in saved_normalized and "runtime" in normalized:
        # Added later to keep wandb-core within the container's PID limit. This
        # changes service resource use only, not the experiment itself.
        saved_runtime = saved_normalized["runtime"]
        current_runtime = normalized["runtime"]
        if (
            "wandb_service_gomaxprocs" not in saved_runtime
            and "wandb_service_gomaxprocs" in current_runtime
        ):
            current_runtime.pop("wandb_service_gomaxprocs")

    # A prepared bundle fixes the actual data. A newer pinned source revision
    # default therefore does not alter the data seen by a resumed run.
    if (
        saved_normalized.get("data_bundle")
        and saved_normalized.get("data_bundle") == normalized.get("data_bundle")
        and saved_normalized.get("data", {}).get("revision") is None
        and normalized.get("data", {}).get("revision") is not None
    ):
        normalized["data"]["revision"] = None

    # Temperature is inert under greedy decoding, so adding this setting to a
    # config created before it existed is safe only when both configs are greedy.
    saved_eval = saved_normalized.get("eval", {})
    current_eval = normalized.get("eval", {})
    if (
        saved_eval.get("greedy") is True
        and current_eval.get("greedy") is True
        and "temperature" not in saved_eval
        and "temperature" in current_eval
    ):
        current_eval.pop("temperature")

    return saved_normalized == normalized


def initialize_wandb(config: DictConfig, run_dir: str | Path, *, job_type: str) -> Any:
    """Create or resume one W&B run, persisting its ID before initialization."""
    _configure_wandb_service_threads(config)
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
    _write_tracking_state(
        state_path,
        {
            "id": run_id,
            "mode": mode,
            "sync_status": "unsynced" if mode == "offline" else "pending",
            "run_status": "running",
        },
    )
    metadata = {
        "resolved_config_path": str(config_path),
        "protocol": "tinymath-r1",
        "resolved_config": sanitized_config(config),
    }
    run = None
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
        # runtime_info probes CUDA. Collect it after W&B has started its service
        # threads so constrained process quotas do not block W&B initialization.
        run.config.update({"runtime": runtime_info().as_dict()}, allow_val_change=True)
        return run
    except (
        wandb.errors.Error,
        requests.exceptions.RequestException,
        OSError,
        ValueError,
        RuntimeError,
    ) as exc:
        if run is not None:
            try:
                finish_wandb(run, exit_code=1)
            except (wandb.errors.Error, requests.exceptions.RequestException, OSError, RuntimeError):
                LOGGER.warning("Could not flush W&B run after initialization failure")
        failed_state = {
            **state,
            "id": run_id,
            "mode": mode,
            "sync_status": "unsynced" if mode == "offline" else "pending",
            "run_status": "failed",
            "exit_code": 1,
        }
        try:
            _write_tracking_state(state_path, failed_state)
        except OSError as state_exc:
            LOGGER.warning("Could not save failed W&B initialization state at %s: %s", state_path, state_exc)
        message = str(exc).replace(os.environ.get("WANDB_API_KEY", "\0"), "<redacted>")
        raise TrackingError(f"W&B initialization failed ({mode} mode): {message}") from None


def initialize_tracked_run(config: DictConfig, run_dir: str | Path, *, job_type: str) -> Any:
    """Start W&B before seeding CUDA, then seed before any training/evaluation work."""
    run = initialize_wandb(config, run_dir, job_type=job_type)
    try:
        seed_everything(int(config.seed), cpu_threads=int(config.runtime.cpu_threads))
    except BaseException:
        finish_wandb(run, exit_code=1)
        raise
    return run


def finish_wandb(run: Any, *, exit_code: int = 0) -> None:
    """Flush a run and record whether its experiment completed successfully."""
    if run is None:
        return
    mode = getattr(getattr(run, "settings", None), "mode", None)
    offline = str(mode) == "offline"
    status = "unsynced" if offline else "pending"
    try:
        # Offline is known not to reach W&B. For online runs, a local status is
        # only marked synced after finish() returns without interruption/error.
        if offline and hasattr(run, "summary"):
            run.summary["tracking/sync_status"] = "unsynced"
        run.finish(exit_code=exit_code)
        if not offline and not bool(getattr(run, "_finish_timed_out", False)):
            status = "synced"
    except BaseException:
        exit_code = 1
        raise
    finally:
        state_path = _ACTIVE_STATE_PATHS.pop(str(getattr(run, "id", "")), None)
        if state_path is not None and state_path.exists():
            state = _tracking_state(state_path)
            state["sync_status"] = status
            state["run_status"] = "finished" if exit_code == 0 else "failed"
            state["exit_code"] = int(exit_code)
            try:
                _write_tracking_state(state_path, state)
            except OSError as exc:
                LOGGER.warning("Could not update W&B resume metadata at %s: %s", state_path, exc)
