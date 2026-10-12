"""Reusable SFT optimization loop over frozen, prepared character data."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from tinymath.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_completion, format_prompt
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.train.sft import build_completion_batch, completion_only_loss


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "prompt": format_prompt(row["question"]),
            "completion": format_completion(row["answer"]),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.randint(len(examples), (batch_size,)).tolist()
    inputs, targets, mask = build_completion_batch(
        [examples[index] for index in indices], tokenizer, context_length=context_length
    )
    return inputs.to(device), targets.to(device), mask.to(device)


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
    train_examples = _examples(read_jsonl(bundle_dir / "sft_train.jsonl"))
    validation_examples = _examples(read_jsonl(bundle_dir / "sft_validation.jsonl"))
    if not train_examples:
        raise ValueError("SFT training split is empty")

    model_values = dict(config.model)
    model_values["vocab_size"] = tokenizer.vocab_size
    model_config = GPTConfig.from_mapping(model_values)
    model = TinyMathGPT(model_config).to(device)
    if config.trainer.bf16 and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.trainer.learning_rate),
        weight_decay=float(config.trainer.weight_decay),
    )
    latest_path = run_dir / "latest.pt"
    start_step = 0
    best_val = math.inf
    tokens_seen = 0
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("tokenizer_sha256") != tokenizer.sha256:
            raise ValueError("Checkpoint tokenizer does not match the prepared data bundle")
        model.load_state_dict(payload["model"])
        is_continuation = str(config.trainer.name) == "continued_sft"
        if payload.get("optimizer") is not None and not is_continuation:
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            tokens_seen = int(payload.get("tokens_seen", 0))
            best_val = float(payload.get("best_val_loss", math.inf))
            if payload.get("rng_state"):
                restore_rng_state(payload["rng_state"])
    elif latest_path.exists():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        tokens_seen = int(payload.get("tokens_seen", 0))
        best_val = float(payload.get("best_val_loss", math.inf))
        restore_rng_state(payload["rng_state"])

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    is_new_metrics = not metrics_path.exists() or start_step == 0
    metrics_file = metrics_path.open("w" if is_new_metrics else "a", newline="", encoding="utf-8")
    metric_writer = csv.DictWriter(
        metrics_file,
        fieldnames=["optimizer_step", "train/loss", "val/loss", "learning_rate", "tokens_seen", "wall_seconds"],
    )
    if is_new_metrics:
        metric_writer.writeheader()
    target_steps = int(config.trainer.max_steps)
    batch_size = int(config.trainer.batch_size)
    accumulation = int(config.trainer.gradient_accumulation)
    eval_every = max(1, int(config.trainer.eval_every))
    save_every = max(1, int(config.trainer.save_every))
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
            amp_enabled = bool(config.trainer.bf16 and device.type == "cuda")
            for _ in range(accumulation):
                inputs, targets, mask = _batch(
                    train_examples,
                    tokenizer,
                    batch_size=batch_size,
                    context_length=model_config.context_length,
                    device=device,
                )
                tokens_this_step += int(mask.sum().item())
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    logits = model(inputs)
                    loss = completion_only_loss(logits, targets, mask) / accumulation
                loss.backward()
                loss_value += float(loss.detach().item()) * accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.trainer.grad_clip))
            optimizer.step()
            tokens_seen += tokens_this_step
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
                    _save_state(best_path=run_dir / "best.pt", model=model, optimizer=optimizer,
                                step=step, tokens_seen=tokens_seen, best_val=best_val,
                                tokenizer=tokenizer, model_config=model_config)
            row = {
                "optimizer_step": step,
                "train/loss": loss_value,
                "val/loss": "" if val_loss is None else val_loss,
                "learning_rate": lr,
                "tokens_seen": tokens_seen,
                "wall_seconds": time.monotonic() - started,
            }
            metric_writer.writerow(row)
            metrics_file.flush()
            if step % int(config.tracking.log_every) == 0 or step == 1 or step == target_steps:
                values = {
                    "optimizer_step": step,
                    "tokens_seen": tokens_seen,
                    "train/loss": loss_value,
                    "train/grad_norm": float(grad_norm.item()),
                    "train/learning_rate": lr,
                    "train/wall_seconds": time.monotonic() - started,
                }
                if val_loss is not None:
                    values["val/loss"] = val_loss
                if device.type == "cuda":
                    values["system/peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                run.log(values, step=step)
            if step % save_every == 0 or step == target_steps:
                _save_state(best_path=latest_path, model=model, optimizer=optimizer,
                            step=step, tokens_seen=tokens_seen, best_val=best_val,
                            tokenizer=tokenizer, model_config=model_config)
    finally:
        metrics_file.close()
    return {
        "step": target_steps,
        "tokens_seen": tokens_seen,
        "best_validation_loss": best_val,
        "metrics_path": str(metrics_path),
        "checkpoint": str(latest_path),
        "parameters": model.count_parameters(),
    }


def _save_state(
    *,
    best_path: Path,
    model: TinyMathGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    best_val: float,
    tokenizer: CharacterTokenizer,
    model_config: GPTConfig,
) -> None:
    save_checkpoint(
        best_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val,
            "rng_state": capture_rng_state(),
            "tokenizer_sha256": tokenizer.sha256,
            "model_config": model_config.__dict__,
        },
    )
