"""Deterministic qualitative samples for native-tokenizer Qwen checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from tinymath.data.hf_reasoning import score_native_reasoning
from tinymath.data.prompts import format_prompt
from tinymath.generation import RolloutConfig
from tinymath.train.hf_generation import (
    generate_native_rollout,
    generate_native_rollout_batch,
)


def select_validation_rows(
    rows: list[dict[str, Any]], *, seed: int, examples_per_module: int
) -> list[dict[str, Any]]:
    """Select a stable, module-balanced sample from parent validation questions."""
    if examples_per_module < 1:
        raise ValueError("examples_per_module must be positive")
    by_module: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("is_sub_task", False):
            continue
        if not row.get("question") or not row.get("answer") or not row.get("module"):
            continue
        by_module.setdefault(str(row["module"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for module, module_rows in sorted(by_module.items()):
        ranked = sorted(
            module_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0validation\0{module}\0{row['question']}".encode("utf-8")
            ).digest(),
        )
        selected.extend(ranked[:examples_per_module])
    return selected


def select_greedy_accuracy_rows(
    rows: list[dict[str, Any]], *, seed: int, max_examples: int | None
) -> list[dict[str, Any]]:
    """Choose a deterministic, module-balanced validation subset (or all rows)."""
    parents = [
        row for row in rows
        if not row.get("is_sub_task", False)
        and row.get("question") and row.get("answer") and row.get("module")
    ]
    if max_examples is None or max_examples >= len(parents):
        return parents
    if max_examples < 1:
        raise ValueError("greedy_accuracy_max_examples must be positive or null")
    by_module: dict[str, list[dict[str, Any]]] = {}
    for row in parents:
        by_module.setdefault(str(row["module"]), []).append(row)
    for module, module_rows in by_module.items():
        module_rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}\0validation-greedy\0{module}\0{row['question']}".encode("utf-8")
            ).digest()
        )
    selected: list[dict[str, Any]] = []
    depth = 0
    modules = sorted(by_module)
    while len(selected) < max_examples:
        added = False
        for module in modules:
            if depth < len(by_module[module]):
                selected.append(by_module[module][depth])
                added = True
                if len(selected) == max_examples:
                    break
        if not added:
            break
        depth += 1
    return selected


@torch.no_grad()
def evaluate_native_greedy_exact_match(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    seed: int,
    context_length: int,
    max_new_tokens: int,
    batch_size: int,
    max_examples: int | None,
    rollout_config: Any,
) -> tuple[float, int]:
    """Measure exact parsed-answer string matches on fixed validation questions."""
    if min(context_length, max_new_tokens, batch_size) < 1:
        raise ValueError("Greedy validation context, completion cap, and batch size must be positive")
    examples = select_greedy_accuracy_rows(
        rows, seed=seed, max_examples=max_examples
    )
    if not examples:
        raise ValueError("No parent validation rows were available for greedy accuracy")
    tokenizer_config = RolloutConfig.from_config(rollout_config)
    matched = 0
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            items = [
                (
                    tokenizer.encode(
                        format_prompt(
                            str(row["question"]), protocol="reasoning_v3"
                        ),
                        add_special_tokens=False,
                    ),
                    row,
                )
                for row in batch
            ]
            rollouts = generate_native_rollout_batch(
                model,
                tokenizer,
                items,
                context_length=context_length,
                max_completion_tokens=max_new_tokens,
                temperature=0.0,
                rollout=tokenizer_config,
                greedy=True,
            )
            for row, rollout in zip(batch, rollouts, strict=True):
                token_ids = list(rollout.token_ids)
                if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                    token_ids.pop()
                completion = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                scored = score_native_reasoning(
                    completion,
                    row,
                    capped=rollout.termination_reason == "budget_exceeded",
                    failure=(
                        rollout.termination_reason
                        if rollout.termination_reason not in {"eos", "budget_exceeded"}
                        else None
                    ),
                    **tokenizer_config.scoring_kwargs(),
                )
                # Exact match is deliberately stricter than mathematical equivalence.
                matched += int(scored.format_valid and scored.answer == str(row["answer"]))
    finally:
        model.train(was_training)
    return matched / len(examples), len(examples)


@torch.no_grad()
def build_native_qualitative_examples(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    seed: int,
    optimizer_step: int,
    context_length: int,
    max_new_tokens: int,
    examples_per_module: int,
    rollout_config: Any,
) -> list[dict[str, Any]]:
    """Greedily generate and score fixed validation examples, including traces."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    examples = select_validation_rows(
        rows, seed=seed, examples_per_module=examples_per_module
    )
    was_training = model.training
    model.eval()
    records: list[dict[str, Any]] = []
    try:
        for row in examples:
            prompt = tokenizer.encode(
                format_prompt(str(row["question"]), protocol="reasoning_v3"),
                add_special_tokens=False,
            )
            room = min(max_new_tokens, context_length - len(prompt))
            if room < 1:
                raise ValueError("Qualitative validation prompt leaves no completion context")
            rollout = generate_native_rollout(
                model,
                tokenizer,
                prompt,
                row,
                context_length=context_length,
                max_completion_tokens=room,
                temperature=0.0,
                rollout=rollout_config,
                greedy=True,
            )
            completion_ids = list(rollout.token_ids)
            if completion_ids and completion_ids[-1] == tokenizer.eos_token_id:
                completion_ids.pop()
            completion = tokenizer.decode(
                completion_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            scored = score_native_reasoning(
                completion,
                row,
                capped=rollout.termination_reason == "budget_exceeded",
                failure=(
                    rollout.termination_reason
                    if rollout.termination_reason not in {"eos", "budget_exceeded"}
                    else None
                ),
                **RolloutConfig.from_config(rollout_config).scoring_kwargs(),
            )
            records.append(
                {
                    "optimizer_step": optimizer_step,
                    "seed": seed,
                    "split": "validation",
                    "module": str(row["module"]),
                    "question": str(row["question"]),
                    "target": str(row["answer"]),
                    "prediction": scored.answer,
                    "completion": completion,
                    "predicted_scratchpad": scored.scratchpad,
                    "reference_scratchpad": str(row.get("scratchpad", "")),
                    "format_valid": scored.format_valid,
                    "trace_valid": scored.trace_valid,
                    "correct": scored.answer_correct,
                    "full_success": scored.full_success,
                    "reward": scored.reward,
                    "termination_reason": rollout.termination_reason,
                    "output_tokens": len(rollout.token_ids),
                    "capped": rollout.termination_reason == "budget_exceeded",
                    "policy_tokens": rollout.policy_tokens,
                    "executor_tokens": rollout.executor_tokens,
                    "executor_calls": len(rollout.call_events),
                }
            )
    finally:
        model.train(was_training)
    return records


def log_native_qualitative_examples(
    *,
    records: list[dict[str, Any]],
    run: Any,
    run_dir: Path,
    wandb_step: int,
) -> Path:
    """Persist a local JSONL snapshot and attach the same rows to the experiment run."""
    if not records:
        raise ValueError("No parent validation rows were available for qualitative logging")
    step = int(records[0]["optimizer_step"])
    output_dir = run_dir / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"step_{step:08d}.jsonl"
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    import wandb

    columns = list(records[0])
    table = wandb.Table(
        columns=columns,
        data=[[record.get(key) for key in columns] for record in records],
    )
    correct = sum(record["correct"] is True for record in records)
    values = {
        "qualitative/examples": table,
        "qualitative/example_count": len(records),
        "qualitative/validation_accuracy": correct / len(records),
        "optimizer_step": step,
    }
    run.log(values, step=wandb_step)
    return destination
