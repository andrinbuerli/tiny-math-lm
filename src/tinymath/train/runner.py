"""Reusable SFT optimization loop over frozen, prepared character data."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from tinymath.checkpoint import (
    capture_rng_state,
    compatibility_metadata,
    restore_rng_state,
    save_checkpoint,
    sha256_file,
    validate_checkpoint_compatibility,
)
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_completion, format_prompt
from tinymath.eval.qualitative import build_qualitative_table
from tinymath.generation import RolloutConfig
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.train.budgets import budget_rows
from tinymath.train.metrics import elapsed_seconds_through_step, truncate_csv_after_step
from tinymath.train.sft import build_completion_batch, completion_only_loss


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sft_training_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep eligible subtask rows while retaining ordinary parent rows."""
    return [
        row for row in rows
        if not row.get("is_sub_task") or row.get("training_eligible") is True
    ]


def _examples(rows: list[dict[str, str]], *, protocol: str = "answer_only") -> list[dict[str, str]]:
    return [
        {
            "prompt": format_prompt(row["question"], protocol=protocol),
            "completion": format_completion(row["answer"]),
            **({"scratchpad": row["scratchpad"]} if protocol in {
                "reasoning_v1", "reasoning_v2", "reasoning_v3", "reasoning_v4"
            } else {}),
            "module": row["module"],
        }
        for row in rows
    ]


def _batch(
    examples: list[dict[str, str]],
    tokenizer: CharacterTokenizer,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(examples), (batch_size,), generator=generator).tolist()
    inputs, targets, mask = build_completion_batch(
        [examples[index] for index in indices], tokenizer, context_length=context_length
    )
    return inputs.to(device), targets.to(device), mask.to(device)


def _bundle_compatibility(
    bundle_dir: Path, tokenizer: CharacterTokenizer, *, context_length: int
) -> dict[str, Any] | None:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        if tokenizer.protocol != "answer_only":
            raise ValueError("Reasoning training requires a provenance-bearing annotated bundle")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("context_length", context_length)) != context_length:
        raise ValueError("Model context length does not match the selected data bundle")
    annotation_sha256 = (manifest.get("annotation") or {}).get("annotation_sha256")
    return compatibility_metadata(
        protocol=tokenizer.protocol,
        annotation_sha256=annotation_sha256,
        tokenizer_sha256=tokenizer.sha256,
        bundle_sha256=sha256_file(manifest_path),
        context_length=context_length,
    )


def _qualitative_rollout_config(config: Any) -> tuple[Any, bool]:
    """Use current logging defaults when a resumed run has an obsolete rollout schema."""
    rollout = getattr(config, "rollout", None)
    if rollout is None:
        return None, False
    try:
        RolloutConfig.from_config(rollout)
    except ValueError as error:
        if str(error) != "Unsupported rollout/scorer/executor/accounting version":
            raise
        return None, True
    return rollout, False


@torch.no_grad()
def _evaluate_loss(
    model: TinyMathGPT,
    examples: list[dict[str, str]],
    tokenizer: CharacterTokenizer,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    bf16: bool,
) -> float:
    if not examples:
        raise ValueError("SFT validation split is empty")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
    with amp:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs, targets, mask = build_completion_batch(batch, tokenizer, context_length=context_length)
            logits = model(inputs.to(device))
            losses.append(completion_only_loss(logits, targets.to(device), mask.to(device)).item())
    model.train(was_training)
    return sum(losses) / len(losses)


def train_sft(
    *,
    config: Any,
    bundle_dir: Path,
    run_dir: Path,
    run: Any,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train to the configured optimizer step and persist resumable state/metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharacterTokenizer.load(bundle_dir / "char_tokenizer.json")
    train_rows = _sft_training_rows(read_jsonl(bundle_dir / "sft_train.jsonl"))
    validation_rows = read_jsonl(bundle_dir / "sft_validation.jsonl")
    training_rows = train_rows
    experiment = getattr(config, "experiment", {})
    bounded_phase = (
        int(experiment.get("number", 0) or 0) == 11
        and str(experiment.get("phase", "main")) in {"smoke", "pilot"}
    )
    if bounded_phase:
        training_rows = budget_rows(
            train_rows,
            examples=int(experiment.get("data_budget", 0)),
            seed=int(config.seed),
        )
        validation_budget = int(experiment.get("validation_examples", len(validation_rows)))
        if validation_budget < len(validation_rows):
            validation_rows = budget_rows(
                validation_rows,
                examples=validation_budget,
                seed=int(config.seed) + 1,
            )
    train_examples = _examples(training_rows, protocol=tokenizer.protocol)
    validation_examples = _examples(validation_rows, protocol=tokenizer.protocol)
    if not train_examples:
        raise ValueError("SFT training split is empty")

    model_values = dict(config.model)
    model_values["vocab_size"] = tokenizer.vocab_size
    model_config = GPTConfig.from_mapping(model_values)
    compatibility = _bundle_compatibility(
        bundle_dir, tokenizer, context_length=model_config.context_length
    )
    model = TinyMathGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.trainer.learning_rate),
        weight_decay=float(config.trainer.weight_decay),
    )
    latest_path = run_dir / "latest.pt"
    start_step = 0
    best_val = math.inf
    tokens_seen = 0
    examples_drawn = 0
    nonpadding_input_tokens = 0
    supervised_tokens = 0
    source_sft_tokens_seen = 0
    source_sft_checkpoint: str | None = None
    data_generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("tokenizer_sha256") != tokenizer.sha256:
            raise ValueError("Checkpoint tokenizer does not match the prepared data bundle")
        if compatibility is not None:
            validate_checkpoint_compatibility(
                payload,
                compatibility,
                allow_legacy_answer_only=(tokenizer.protocol == "answer_only"),
            )
        model.load_state_dict(payload["model"])
        is_continuation = str(config.trainer.name) == "continued_sft"
        if is_continuation:
            source_sft_checkpoint = str(init_checkpoint.resolve())
            source_sft_tokens_seen = int(
                payload.get("source_sft_tokens_seen", 0)
                if payload.get("training_arm") == "continued_sft"
                else payload.get("tokens_seen", 0)
            )
            if latest_path.exists():
                resume_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
                if resume_payload.get("tokenizer_sha256") != tokenizer.sha256:
                    raise ValueError(
                        "Continued-SFT checkpoint tokenizer does not match the data bundle"
                    )
                if compatibility is not None:
                    validate_checkpoint_compatibility(
                        resume_payload,
                        compatibility,
                        allow_legacy_answer_only=(tokenizer.protocol == "answer_only"),
                    )
                saved_source = resume_payload.get("source_sft_checkpoint")
                if saved_source and Path(saved_source).resolve() != init_checkpoint.resolve():
                    raise ValueError(
                        "Continued-SFT checkpoint came from a different SFT checkpoint"
                    )
                if resume_payload.get("optimizer") is None:
                    raise ValueError("Continued-SFT resume checkpoint has no optimizer state")
                model.load_state_dict(resume_payload["model"])
                optimizer.load_state_dict(resume_payload["optimizer"])
                start_step = int(resume_payload["step"])
                tokens_seen = int(resume_payload.get("tokens_seen", 0))
                examples_drawn = int(resume_payload.get("examples_drawn", 0))
                nonpadding_input_tokens = int(resume_payload.get("nonpadding_input_tokens", 0))
                supervised_tokens = int(resume_payload.get("supervised_tokens", 0))
                source_sft_tokens_seen = int(
                    resume_payload.get("source_sft_tokens_seen", source_sft_tokens_seen)
                )
                best_val = float(resume_payload.get("best_val_loss", math.inf))
                best_val = _latest_best_val(run_dir, best_val)
                restore_rng_state(resume_payload["rng_state"])
                if resume_payload.get("data_generator_state") is not None:
                    data_generator.set_state(resume_payload["data_generator_state"])
                source_sft_checkpoint = saved_source or source_sft_checkpoint
        elif payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            tokens_seen = int(payload.get("tokens_seen", 0))
            examples_drawn = int(payload.get("examples_drawn", 0))
            nonpadding_input_tokens = int(payload.get("nonpadding_input_tokens", 0))
            supervised_tokens = int(payload.get("supervised_tokens", 0))
            source_sft_tokens_seen = int(payload.get("source_sft_tokens_seen", 0))
            best_val = float(payload.get("best_val_loss", math.inf))
            source_sft_checkpoint = payload.get("source_sft_checkpoint")
            if payload.get("rng_state"):
                restore_rng_state(payload["rng_state"])
            if payload.get("data_generator_state") is not None:
                data_generator.set_state(payload["data_generator_state"])
    elif latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        if compatibility is not None:
            validate_checkpoint_compatibility(
                payload,
                compatibility,
                allow_legacy_answer_only=(tokenizer.protocol == "answer_only"),
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        tokens_seen = int(payload.get("tokens_seen", 0))
        examples_drawn = int(payload.get("examples_drawn", 0))
        nonpadding_input_tokens = int(payload.get("nonpadding_input_tokens", 0))
        supervised_tokens = int(payload.get("supervised_tokens", 0))
        source_sft_tokens_seen = int(payload.get("source_sft_tokens_seen", 0))
        best_val = float(payload.get("best_val_loss", math.inf))
        best_val = _latest_best_val(run_dir, best_val)
        source_sft_checkpoint = payload.get("source_sft_checkpoint")
        restore_rng_state(payload["rng_state"])
        if payload.get("data_generator_state") is not None:
            data_generator.set_state(payload["data_generator_state"])

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    elapsed_before = (
        elapsed_seconds_through_step(metrics_path, step_column="optimizer_step", step=start_step)
        if start_step
        else 0.0
    )
    if start_step:
        truncate_csv_after_step(metrics_path, step_column="optimizer_step", step=start_step)
    is_new_metrics = not metrics_path.exists() or start_step == 0
    metrics_file = metrics_path.open("w" if is_new_metrics else "a", newline="", encoding="utf-8")
    metric_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "optimizer_step", "train/loss", "val/loss", "learning_rate", "tokens_seen",
            "examples_drawn", "nonpadding_input_tokens", "padded_compute_positions",
            "supervised_tokens", "additional_estimated_flops",
            "tokens_per_second", "estimated_flops", "wall_seconds", "train/greedy_accuracy",
            "val/greedy_accuracy", "test/greedy_accuracy",
        ],
    )
    if is_new_metrics:
        metric_writer.writeheader()
    target_steps = int(config.trainer.max_steps)
    batch_size = int(config.trainer.batch_size)
    accumulation = int(config.trainer.gradient_accumulation)
    eval_every = max(1, int(config.trainer.eval_every))
    save_every = max(1, int(config.trainer.save_every))
    qualitative_every = max(1, int(getattr(getattr(config, "qualitative", None), "every_steps", 20_000)))
    qualitative_count = 0
    qualitative_rollout, legacy_rollout_fallback = _qualitative_rollout_config(config)
    define_metric = getattr(run, "define_metric", None)
    if callable(define_metric):
        define_metric("optimizer_step")
        for metric in ("train/*", "val/*", "test/*", "system/*", "qualitative/*"):
            define_metric(metric, step_metric="optimizer_step")
    if start_step > 0:
        qualitative_table, qualitative_count, greedy_accuracy = build_qualitative_table(
            model=model,
            tokenizer=tokenizer,
            split_rows={
                "train": train_rows,
                "validation": validation_rows,
                "test": read_jsonl(bundle_dir / "test.jsonl"),
            },
            training_arm=str(config.trainer.name),
            seed=int(config.seed),
            max_new_tokens=int(getattr(getattr(config, "eval", None), "max_new_tokens", 128)),
            rollout=qualitative_rollout,
            examples_per_module=int(
                getattr(getattr(config, "qualitative", None), "examples_per_module", 2)
            ),
        )
        resume_values = {
            "optimizer_step": start_step,
            "qualitative/examples": qualitative_table,
            "qualitative/resume_checkpoint": True,
        }
        if legacy_rollout_fallback:
            resume_values["qualitative/legacy_rollout_fallback"] = True
        for split, value in greedy_accuracy.items():
            metric_name = "val/greedy_accuracy" if split == "validation" else f"{split}/greedy_accuracy"
            resume_values[metric_name] = value
        summary = getattr(run, "summary", None)
        if summary is not None:
            summary["qualitative/example_count"] = qualitative_count
            summary["qualitative/splits"] = ["train", "validation", "test"]
            summary["training/latest_inference_checkpoint"] = str(
                run_dir / "latest_inference.pt"
            )
        run.log(resume_values)
    warmup = int(config.trainer.warmup_steps)
    base_lr = float(config.trainer.learning_rate)
    started = time.monotonic()
    model.train()
    try:
        for step in range(start_step + 1, target_steps + 1):
            if warmup and step <= warmup:
                lr_scale = step / warmup
            else:
                progress = (step - warmup) / max(1, target_steps - warmup)
                lr_scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            lr = base_lr * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            tokens_this_step = 0
            examples_this_step = batch_size * accumulation
            nonpadding_tokens_this_step = 0
            supervised_tokens_this_step = 0
            amp_enabled = bool(config.trainer.bf16 and device.type == "cuda")
            for _ in range(accumulation):
                inputs, targets, mask = _batch(
                    train_examples,
                    tokenizer,
                    batch_size=batch_size,
                    context_length=model_config.context_length,
                    device=device,
                    generator=data_generator,
                )
                # Count all model input positions, including prompt and padding compute.
                tokens_this_step += inputs.numel()
                nonpadding_tokens_this_step += int(inputs.ne(tokenizer.pad_id).sum().item())
                supervised_tokens_this_step += int(mask.sum().item())
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = model(inputs)
                    loss = completion_only_loss(logits, targets, mask) / accumulation
                loss.backward()
                loss_value += float(loss.detach().item()) * accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.trainer.grad_clip))
            optimizer.step()
            tokens_seen += tokens_this_step
            examples_drawn += examples_this_step
            nonpadding_input_tokens += nonpadding_tokens_this_step
            supervised_tokens += supervised_tokens_this_step
            val_loss: float | None = None
            if step % eval_every == 0 or step == target_steps:
                val_loss = _evaluate_loss(
                    model,
                    validation_examples,
                    tokenizer,
                    batch_size=batch_size,
                    context_length=model_config.context_length,
                    device=device,
                    bf16=amp_enabled,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    _save_state(
                        best_path=run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        tokens_seen=tokens_seen,
                        examples_drawn=examples_drawn,
                        nonpadding_input_tokens=nonpadding_input_tokens,
                        supervised_tokens=supervised_tokens,
                        source_sft_tokens_seen=source_sft_tokens_seen,
                        best_val=best_val,
                        tokenizer=tokenizer,
                        model_config=model_config,
                        compatibility=compatibility,
                        data_generator_state=data_generator.get_state(),
                        source_sft_checkpoint=source_sft_checkpoint,
                        training_arm=str(config.trainer.name),
                    )
            elapsed = elapsed_before + time.monotonic() - started
            row = {
                "optimizer_step": step,
                "train/loss": loss_value,
                "val/loss": "" if val_loss is None else val_loss,
                "learning_rate": lr,
                "tokens_seen": tokens_seen,
                "examples_drawn": examples_drawn,
                "nonpadding_input_tokens": nonpadding_input_tokens,
                "padded_compute_positions": tokens_seen,
                "supervised_tokens": supervised_tokens,
                "tokens_per_second": tokens_seen / max(1e-9, elapsed),
                "estimated_flops": 6 * model.count_parameters() * tokens_seen,
                "additional_estimated_flops": (
                    6 * model.count_parameters() * max(0, tokens_seen - source_sft_tokens_seen)
                ),
                "wall_seconds": elapsed,
            }
            values = {
                "optimizer_step": step,
                "tokens_seen": tokens_seen,
                "train/examples_drawn": examples_drawn,
                "train/nonpadding_input_tokens": nonpadding_input_tokens,
                "train/padded_compute_positions": tokens_seen,
                "train/supervised_tokens": supervised_tokens,
                "train/loss": loss_value,
                "train/grad_norm": float(grad_norm.item()),
                "train/learning_rate": lr,
                "train/wall_seconds": elapsed,
                "train/tokens_per_second": tokens_seen / max(1e-9, elapsed),
                "train/estimated_flops": 6 * model.count_parameters() * tokens_seen,
                "train/additional_estimated_flops": (
                    6 * model.count_parameters() * max(0, tokens_seen - source_sft_tokens_seen)
                ),
            }
            if val_loss is not None:
                values["val/loss"] = val_loss
            if device.type == "cuda":
                values["system/peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            if step % save_every == 0 or step == target_steps:
                _save_state(
                    best_path=latest_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    tokens_seen=tokens_seen,
                    examples_drawn=examples_drawn,
                    nonpadding_input_tokens=nonpadding_input_tokens,
                    supervised_tokens=supervised_tokens,
                    source_sft_tokens_seen=source_sft_tokens_seen,
                    best_val=best_val,
                    tokenizer=tokenizer,
                    model_config=model_config,
                    compatibility=compatibility,
                    data_generator_state=data_generator.get_state(),
                    source_sft_checkpoint=source_sft_checkpoint,
                    training_arm=str(config.trainer.name),
                )
                _save_inference_state(
                    run_dir / "latest_inference.pt",
                    model=model,
                    step=step,
                    tokenizer=tokenizer,
                    model_config=model_config,
                    compatibility=compatibility,
                    training_arm=str(config.trainer.name),
                    source_sft_checkpoint=source_sft_checkpoint,
                )
            qualitative_due = step % qualitative_every == 0 or step == target_steps
            if qualitative_due:
                qualitative_table, qualitative_count, greedy_accuracy = build_qualitative_table(
                    model=model,
                    tokenizer=tokenizer,
                    split_rows={
                        "train": train_rows,
                        "validation": validation_rows,
                        "test": read_jsonl(bundle_dir / "test.jsonl"),
                    },
                    training_arm=str(config.trainer.name),
                    seed=int(config.seed),
                    max_new_tokens=int(
                        getattr(getattr(config, "eval", None), "max_new_tokens", 128)
                    ),
                    rollout=qualitative_rollout,
                    examples_per_module=int(
                        getattr(getattr(config, "qualitative", None), "examples_per_module", 2)
                    ),
                )
                summary = getattr(run, "summary", None)
                if summary is not None:
                    summary["qualitative/example_count"] = qualitative_count
                    summary["qualitative/splits"] = ["train", "validation", "test"]
                    summary["training/latest_inference_checkpoint"] = str(
                        run_dir / "latest_inference.pt"
                    )
                values["qualitative/examples"] = qualitative_table
                if legacy_rollout_fallback:
                    values["qualitative/legacy_rollout_fallback"] = True
                for split, value in greedy_accuracy.items():
                    metric_name = "val/greedy_accuracy" if split == "validation" else f"{split}/greedy_accuracy"
                    values[metric_name] = value
                    row[metric_name] = value
            metric_writer.writerow(row)
            metrics_file.flush()
            if (
                step % int(config.tracking.log_every) == 0
                or step == 1
                or step == target_steps
                or qualitative_due
            ):
                run.log(values)
    finally:
        metrics_file.close()
    return {
        "step": target_steps,
        "tokens_seen": tokens_seen,
        "examples_drawn": examples_drawn,
        "nonpadding_input_tokens": nonpadding_input_tokens,
        "padded_compute_positions": tokens_seen,
        "supervised_tokens": supervised_tokens,
        "additional_estimated_flops": (
            6 * model.count_parameters() * max(0, tokens_seen - source_sft_tokens_seen)
        ),
        "training_examples": len(train_examples),
        "best_validation_loss": best_val,
        "metrics_path": str(metrics_path),
        "checkpoint": str(latest_path),
        "inference_checkpoint": str(run_dir / "latest_inference.pt"),
        "parameters": model.count_parameters(),
        "qualitative_examples": qualitative_count,
        "source_sft_checkpoint": source_sft_checkpoint,
    }


def _save_state(
    *,
    best_path: Path,
    model: TinyMathGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    examples_drawn: int,
    nonpadding_input_tokens: int,
    supervised_tokens: int,
    source_sft_tokens_seen: int,
    best_val: float,
    tokenizer: CharacterTokenizer,
    model_config: GPTConfig,
    compatibility: dict[str, Any] | None,
    data_generator_state: torch.Tensor,
    source_sft_checkpoint: str | None,
    training_arm: str,
) -> None:
    save_checkpoint(
        best_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "examples_drawn": examples_drawn,
            "nonpadding_input_tokens": nonpadding_input_tokens,
            "padded_compute_positions": tokens_seen,
            "supervised_tokens": supervised_tokens,
            "source_sft_tokens_seen": source_sft_tokens_seen,
            "best_val_loss": best_val,
            "rng_state": capture_rng_state(),
            "data_generator_state": data_generator_state,
            "tokenizer_sha256": tokenizer.sha256,
            "model_config": model_config.__dict__,
            "source_sft_checkpoint": source_sft_checkpoint,
            "training_arm": training_arm,
            **({"compatibility": compatibility} if compatibility is not None else {}),
        },
    )


def _latest_best_val(run_dir: Path, fallback: float) -> float:
    """Keep the validation best if it advanced after the last resumable checkpoint."""
    best_path = run_dir / "best.pt"
    if not best_path.is_file():
        return fallback
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    return min(fallback, float(best_payload.get("best_val_loss", math.inf)))


def _save_inference_state(
    path: Path,
    *,
    model: TinyMathGPT,
    step: int,
    tokenizer: CharacterTokenizer,
    model_config: GPTConfig,
    compatibility: dict[str, Any] | None,
    training_arm: str,
    source_sft_checkpoint: str | None,
) -> None:
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "step": step,
            "tokenizer_sha256": tokenizer.sha256,
            "model_config": model_config.__dict__,
            "training_arm": training_arm,
            "source_sft_checkpoint": source_sft_checkpoint,
            **({"compatibility": compatibility} if compatibility is not None else {}),
        },
    )
