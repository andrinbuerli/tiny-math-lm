"""Stable W&B identity construction for Experiment 11 jobs.

The helper is deliberately side-effect free. Call it with the resolved Hydra
configuration and the metadata for the concrete job before ``wandb.init``.
Historical experiments keep the names already resolved by their configs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

EXP11_NUMBER = 11
EXP11_CANONICAL_NAME = "exp11_reasoning_500k_100k"
EXP11_SLUG = "exp11-reasoning-500k-100k"
EXP11_TITLE = "Experiment 11 — Compact handwritten reasoning, 500k examples, 100k updates"
EXP11_BASELINE = "restarted_100k_greedy_restart_2"


def _plain_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        result = OmegaConf.to_container(config, resolve=True)
    else:
        result = dict(config)
    if not isinstance(result, dict):  # pragma: no cover - protects the public API
        raise TypeError("configuration must resolve to a mapping")
    return result


def _slug(value: Any, *, field: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    if not value:
        raise ValueError(f"{field} is required to build a stable Experiment 11 name")
    return value


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def build_run_naming(
    config: Mapping[str, Any] | DictConfig,
    *,
    job_type: str,
    phase: str | None = None,
    seed: int | None = None,
    model: str | None = None,
    arm: str | None = None,
    source_arm: str | None = None,
    examples: int | None = None,
    updates: int | None = None,
    checkpoint_selector: str | None = None,
    checkpoint_step: int | str | None = None,
    checkpoint_hash: str | None = None,
) -> dict[str, Any]:
    """Return W&B project/group/name plus sanitized Experiment 11 lineage.

    ``job_type`` identifies the concrete invocation (for example ``training``,
    ``evaluation``, ``annotation``, ``smoke``, ``pilot``, ``qualitative``, or
    ``comparison``). Optional identity fields override corresponding values in
    the resolved config because a single config can launch several distinct
    jobs. Evaluation names require checkpoint selector, step, and hash so two
    checkpoints can never silently share a human-readable identity.
    """
    cfg = _plain_config(config)
    experiment = cfg.get("experiment", {}) or {}
    tracking = cfg.get("tracking", {}) or {}
    is_exp11 = int(experiment.get("number", 0) or 0) == EXP11_NUMBER or (
        experiment.get("name") == EXP11_CANONICAL_NAME
    )

    # Experiments 01–10 retain their existing group/name and therefore their
    # resume identities. This function does not mint IDs; W&B's persisted ID
    # remains authoritative on resume.
    project = str(tracking.get("project", "tinymath-rl"))
    if not is_exp11:
        return {
            "project": project,
            "group": str(tracking.get("group", experiment.get("name", "default"))),
            "name": str(tracking.get("name", "run")),
            "metadata": {},
        }

    seed = int(_first(seed, experiment.get("seed"), cfg.get("seed"), default=18))
    actual_model = _first(model, (cfg.get("model") or {}).get("name"), default="unknown-model")
    actual_arm = _first(arm, source_arm, phase, default="unknown-arm")
    actual_phase = _first(phase, job_type, default="training")
    annotation_rows = _first(experiment.get("annotation_rows"), 607000)
    actual_examples = _first(
        examples,
        annotation_rows
        if str(job_type).lower().replace("_", "-") in {"annotation", "data-annotation"}
        else None,
        experiment.get("examples"),
        experiment.get("data_budget"),
        default=0,
    )
    actual_updates = _first(
        updates,
        experiment.get("updates"),
        experiment.get("max_steps"),
        default=0,
    )
    model_slug = _slug(actual_model, field="model")
    arm_slug = _slug(actual_arm, field="arm")
    kind = _slug(job_type, field="job_type")
    group = f"{EXP11_SLUG}-seed{seed}"

    if kind in {"training", "train", "sft", "grpo", "continued-sft", "continued_sft"}:
        # For training, phase/arm names the specific algorithmic arm.
        name = f"{EXP11_SLUG}-main-{model_slug}-{arm_slug}-seed{seed}"
    elif kind in {"evaluation", "eval"}:
        selected_arm = _slug(_first(source_arm, arm, default="unknown-arm"), field="source_arm")
        if checkpoint_selector is None or checkpoint_selector == "":
            raise ValueError("checkpoint_selector is required for an Experiment 11 evaluation name")
        selector = _slug(checkpoint_selector, field="checkpoint_selector")
        step = _first(checkpoint_step)
        if step is None:
            raise ValueError("checkpoint_step is required for an Experiment 11 evaluation name")
        digest = _first(checkpoint_hash, default="")
        if not digest:
            raise ValueError("checkpoint_hash is required for an Experiment 11 evaluation name")
        hash_identity = _slug(str(digest)[:12], field="checkpoint_hash")
        name = (
            f"{EXP11_SLUG}-evaluation-{model_slug}-{selected_arm}-"
            f"{selector}-step{_slug(step, field='checkpoint_step')}-{hash_identity}-seed{seed}"
        )
    elif kind in {"annotation", "data-annotation", "data_annotation"}:
        name = f"{EXP11_SLUG}-data-annotation-{int(actual_examples)}-rows-seed{seed}"
    elif kind in {"comparison", "compare"}:
        name = f"{EXP11_SLUG}-comparison-vs-exp10-seed{seed}"
    elif kind in {"smoke", "pilot"}:
        # Budgets are always explicit in these isolated optimization runs.
        if int(actual_examples) <= 0 or int(actual_updates) <= 0:
            raise ValueError("positive example and update budgets are required for smoke/pilot names")
        name = (
            f"{EXP11_SLUG}-{kind}-{int(actual_examples)}-examples-"
            f"{int(actual_updates)}-updates-{model_slug}-{arm_slug}-seed{seed}"
        )
    elif kind in {"qualitative", "qual"}:
        name = f"{EXP11_SLUG}-qualitative-{model_slug}-{arm_slug}-seed{seed}"
    else:
        name = f"{EXP11_SLUG}-{kind}-{model_slug}-{arm_slug}-seed{seed}"

    context = int(
        _first(
            (cfg.get("model") or {}).get("context_length"),
            experiment.get("context_length"),
            default=512,
        )
    )
    metadata = {
        "experiment_number": EXP11_NUMBER,
        "experiment_name": EXP11_CANONICAL_NAME,
        "experiment_title": EXP11_TITLE,
        "phase": str(actual_phase),
        "examples": int(actual_examples),
        "updates": int(actual_updates),
        "model": str(actual_model),
        "arm": str(actual_arm),
        "protocol": str(experiment.get("protocol", "reasoning_v1")),
        "context": context,
        "seed": seed,
        "baseline_identity": str(experiment.get("baseline", EXP11_BASELINE)),
        "lineage": str(experiment.get("lineage", EXP11_BASELINE)),
    }
    return {"project": project, "group": group, "name": name, "metadata": metadata}
