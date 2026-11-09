"""Deterministic, module-balanced row selection for bounded optimization runs."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def budget_rows(
    rows: list[dict[str, Any]], *, examples: int, seed: int
) -> list[dict[str, Any]]:
    """Select up to ``examples`` rows evenly by module, preserving source order."""
    if examples < 1:
        raise ValueError("The training example budget must be positive")
    if examples >= len(rows):
        return rows
    by_module: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        module = row.get("module")
        if not isinstance(module, str) or not module:
            raise ValueError(f"Training row {index} has no module identity")
        by_module[module].append(index)
    modules = sorted(by_module)
    if not modules:
        raise ValueError("Cannot select a training budget from an empty pool")

    quotas = {module: examples // len(modules) for module in modules}
    for module in modules[: examples % len(modules)]:
        quotas[module] += 1
    remaining = 0
    for module in modules:
        if quotas[module] > len(by_module[module]):
            remaining += quotas[module] - len(by_module[module])
            quotas[module] = len(by_module[module])
    while remaining:
        available = [module for module in modules if quotas[module] < len(by_module[module])]
        if not available:
            break
        for module in available:
            if not remaining:
                break
            quotas[module] += 1
            remaining -= 1

    rng = random.Random(seed)
    selected: set[int] = set()
    for module in modules:
        selected.update(rng.sample(by_module[module], quotas[module]))
    if len(selected) != min(examples, len(rows)):
        raise AssertionError("Module-balanced sampler did not meet its example budget")
    return [row for index, row in enumerate(rows) if index in selected]


def isolated_training_run_dir(
    config: Mapping[str, Any], *, model: str, arm: str, runs_root: str = "/workspace/runs"
) -> str:
    """Return historical main paths or budget-specific Experiment 11 paths."""
    experiment = config.get("experiment", {})
    experiment_number = int(experiment.get("number", 0) or 0)
    phase = str(experiment.get("phase", "main"))
    base = f"{runs_root}/{experiment['name']}/{model}/{arm}"
    if experiment_number == 11 and phase in {"smoke", "pilot"}:
        examples = int(experiment.get("data_budget", 0))
        updates = int(experiment.get("updates", experiment.get("max_steps", 0)))
        if examples < 1 or updates < 1:
            raise ValueError("Smoke/pilot runs require positive example and update budgets")
        base += f"/{phase}-{examples}-examples-{updates}-updates"
    return f"{base}/seed{int(config.get('seed', experiment.get('seed', 0)))}"
