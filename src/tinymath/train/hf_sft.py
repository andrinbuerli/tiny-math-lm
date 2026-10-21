"""Full-parameter completion-only SFT for a pinned Hugging Face causal LM."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tinymath.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from tinymath.data.hf_cached_dataset import HFTokenSplit
from tinymath.data.hf_reasoning import sha256_file
from tinymath.model.hf_qwen import completion_loss, load_model, load_tokenizer, save_model_identity


def _find_cache(config: Any) -> tuple[Path, dict[str, Any]]:
    cache_root = Path("/workspace/data/cache/tokenized")
    revision = str(config.model.revision)
    candidates = sorted(cache_root.glob(f"exp16_qwen3_base_reasoning_v3-{revision[:12]}/*/manifest.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one cache for pinned tokenizer, found {len(candidates)}")
    manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tokenizer_revision") != revision:
        raise ValueError("Token cache tokenizer revision does not match the selected model")
    expected_bundle = str(config.checkpoint_provenance.data_bundle_manifest_sha256)
    if manifest.get("source_bundle_manifest_sha256") != expected_bundle:
        raise ValueError("Token cache was created from another reasoning-v3 bundle")
    return manifest_path.parent, manifest


def _pilot_indices(split: HFTokenSplit, row_cap: int, seed: int) -> list[int]:
    """Choose a deterministic pilot subset that always contains the longest row."""
    if row_cap < 1:
        raise ValueError("pilot_train_rows must be positive")
    if row_cap >= len(split):
        return list(range(len(split)))
    lengths = np.diff(split.offsets)
    longest = int(np.argmax(lengths))
    candidates = np.concatenate((np.arange(longest), np.arange(longest + 1, len(split))))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(candidates, size=row_cap - 1, replace=False).tolist()
    return sorted([longest, *(int(index) for index in sampled)])


def _plan_compute_matched_updates(
    split: HFTokenSplit,
    train_indices: list[int],
    *,
    seed: int,
    parameters: int,
    target_flops: int,
    batch_size: int,
    accumulation: int,
) -> tuple[int, int, int]:
    """Plan the first full optimizer boundary reaching the 6PN supervised-FLOP target."""
    if min(parameters, target_flops, batch_size, accumulation) < 1 or not train_indices:
        raise ValueError("Compute matching requires positive parameters, budget, and batches")
    target_tokens = math.ceil(target_flops / (6 * parameters))
    lengths = np.diff(split.offsets)
    total_tokens = updates = epochs = 0
    while total_tokens < target_tokens:
        generator = torch.Generator(device="cpu").manual_seed(seed + epochs)
        shuffled = torch.randperm(len(train_indices), generator=generator).tolist()
        order = [train_indices[index] for index in shuffled]
        accumulated_rows = accumulated_microbatches = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            accumulated_rows += sum(int(lengths[index]) for index in batch)
            accumulated_microbatches += 1
            if accumulated_microbatches < accumulation and start + batch_size < len(order):
                continue
            updates += 1
            total_tokens += accumulated_rows
            accumulated_rows = accumulated_microbatches = 0
            if total_tokens >= target_tokens:
                epochs += 1
                return updates, total_tokens, epochs
        epochs += 1
        if epochs > 10000:
            raise RuntimeError("Compute-matched control schedule exceeds 10,000 data epochs")
    return updates, total_tokens, epochs


def _batch_loss(
    model: Any,
    split: HFTokenSplit,
    indices: list[int],
    *,
    pad_id: int,
    device: torch.device,
    backward: bool,
) -> tuple[float, int, int]:
    input_ids, attention_mask, labels = split.batch(indices, pad_id=pad_id, device=device)
    loss_sum, token_count = completion_loss(model, input_ids, attention_mask, labels)
    if backward:
        loss_sum.backward()
    input_tokens = int(attention_mask.sum().item())
    return float(loss_sum.detach().item()), token_count, input_tokens


@torch.no_grad()
def _evaluate_loss(
    model: Any,
    split: HFTokenSplit,
    *,
    pad_id: int,
    device: torch.device,
    batch_size: int,
    max_examples: int | None = None,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    count = len(split) if max_examples is None else min(len(split), max_examples)
    was_training = model.training
    model.eval()
    for start in range(0, count, batch_size):
        indices = list(range(start, min(start + batch_size, count)))
        loss_sum, tokens, _input_tokens = _batch_loss(
            model, split, indices, pad_id=pad_id, device=device, backward=False
        )
        total_loss += loss_sum
        total_tokens += tokens
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Validation has no completion tokens")
    return total_loss / total_tokens


def _cosine_lr(step: int, *, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    schedule_step = step + 1
    if warmup_steps and schedule_step <= warmup_steps:
        return base_lr * float(schedule_step) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((schedule_step - warmup_steps) / decay_steps, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    step: int,
    epoch: int,
    cursor: int,
    best_val: float,
    epoch_best_before: float,
    no_improvement_epochs: int,
    tokens_seen: int,
    examples_seen: int,
    supervised_tokens_seen: int,
    peak_gpu_memory_bytes: int,
    wall_seconds: float,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "format": "exp16_hf_sft_v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "cursor": cursor,
        "best_val_loss": best_val,
        "epoch_best_before": epoch_best_before,
        "no_improvement_epochs": no_improvement_epochs,
        "tokens_seen": tokens_seen,
        "supervised_tokens_seen": supervised_tokens_seen,
        "examples_seen": examples_seen,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "wall_seconds": wall_seconds,
        "config_sha256": config_sha256,
        "rng_state": capture_rng_state(),
    }


def _write_metric(writer: csv.DictWriter, handle: Any, values: dict[str, Any]) -> None:
    writer.writerow(values)
    handle.flush()


def train_hf_sft(
    *,
    config: Any,
    run_dir: Path,
    run: Any,
    init_checkpoint: Path | None = None,
    device_override: torch.device | None = None,
) -> dict[str, Any]:
    """Train or resume a 3-epoch full-parameter Qwen3 SFT run."""
    cache_dir, cache_manifest = _find_cache(config)
    train_split = HFTokenSplit(cache_dir, "train")
    validation_split = HFTokenSplit(cache_dir, "validation")
    experiment_phase = str(config.experiment.phase)
    if experiment_phase in {"pilot", "smoke"}:
        row_cap = int(config.experiment.get("pilot_train_rows", 64))
        train_indices = _pilot_indices(train_split, row_cap, int(config.seed))
        validation_cap = int(config.experiment.get("pilot_validation_rows", 64))
    else:
        train_indices = list(range(len(train_split)))
        validation_cap = None

    device = device_override or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_override is None and device.type != "cuda":
        raise RuntimeError("Experiment 16 optimization requires the admitted RTX 4090")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Experiment 16 requires bf16 support on the admitted CUDA device")
    model = load_model(config.model, checkpoint=init_checkpoint, training=True).to(device)
    if int(config.model.context_length) != 512:
        raise ValueError("Experiment 16 context length is fixed at 512")
    pad_id = int(cache_manifest["tokenizer_identity"]["pad_token_id"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if hasattr(run, "config"):
        run.config.update(
            {
                "model_parameters": parameters,
                "model_revision": str(config.model.revision),
                "tokenizer_sha256": cache_manifest["tokenizer_sha256"],
                "model_config_sha256": cache_manifest["model_config_sha256"],
                "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                "sft_training_rows": len(train_indices),
                "sft_validation_rows": len(validation_split),
                "source_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            },
            allow_val_change=True,
        )

    base_lr = float(config.trainer.learning_rate)
    weight_decay = float(config.trainer.weight_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        foreach=False,
    )
    batch_size = int(config.trainer.batch_size)
    accumulation = int(config.trainer.gradient_accumulation)
    max_epochs = int(config.trainer.max_epochs)
    max_steps = int(config.trainer.max_steps)
    eval_every = int(config.trainer.eval_every)
    save_every = int(config.trainer.save_every)
    warmup_ratio = float(config.trainer.warmup_ratio)
    updates_per_epoch = math.ceil(len(train_indices) / (batch_size * accumulation))
    target_flops_value = config.trainer.get("compute_budget_flops")
    target_flops = int(target_flops_value) if target_flops_value is not None else None
    planned_tokens = None
    planned_epochs = None
    trainer_name = str(config.trainer.get("name", "sft"))
    if trainer_name == "continued_sft":
        if target_flops is None:
            raise ValueError("Continued SFT requires trainer.compute_budget_flops from completed GRPO")
        total_updates, planned_tokens, planned_epochs = _plan_compute_matched_updates(
            train_split,
            train_indices,
            seed=int(config.seed),
            parameters=parameters,
            target_flops=target_flops,
            batch_size=batch_size,
            accumulation=accumulation,
        )
        if total_updates > max_steps:
            raise ValueError(
                f"Continued-SFT target needs {total_updates} optimizer updates, above "
                f"trainer.max_steps={max_steps}"
            )
        max_epochs = max(max_epochs, int(planned_epochs))
    else:
        total_updates = min(updates_per_epoch * max_epochs, max_steps)
    warmup_steps = math.ceil(total_updates * warmup_ratio)
    if hasattr(run, "config"):
        run.config.update(
            {
                "compute_match.target_flops": target_flops,
                "compute_match.planned_optimizer_steps": (
                    total_updates if target_flops is not None else None
                ),
                "compute_match.planned_input_tokens": planned_tokens,
                "compute_match.planned_flops": (
                    6 * parameters * planned_tokens if planned_tokens is not None else None
                ),
                "compute_match.planned_epochs": planned_epochs,
                "compute_match.warmup_steps": warmup_steps,
                "compute_match.estimator": "6*parameters*nonpadding_input_tokens",
            },
            allow_val_change=True,
        )
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "trainer": dict(config.trainer),
                "model": dict(config.model),
                "seed": int(config.seed),
                "phase": experiment_phase,
                "source_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    step = epoch = cursor = 0
    best_val = math.inf
    epoch_best_before = math.inf
    no_improvement_epochs = 0
    tokens_seen = supervised_tokens_seen = examples_seen = 0
    wall_seconds_before = 0.0
    peak_memory_seen = 0
    if latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state.get("format") != "exp16_hf_sft_v1" or state.get("config_sha256") != config_hash:
            raise ValueError("Experiment 16 SFT resume checkpoint identity/configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, cursor = int(state["step"]), int(state["epoch"]), int(state["cursor"])
        best_val = float(state["best_val_loss"])
        epoch_best_before = float(state.get("epoch_best_before", best_val))
        no_improvement_epochs = int(state["no_improvement_epochs"])
        tokens_seen = int(state["tokens_seen"])
        supervised_tokens_seen = int(state.get("supervised_tokens_seen", tokens_seen))
        examples_seen = int(state["examples_seen"])
        peak_memory_seen = int(state.get("peak_gpu_memory_bytes", 0))
        wall_seconds_before = float(state.get("wall_seconds", 0.0))
        restore_rng_state(state["rng_state"])
    fields = [
        "optimizer_step", "epoch", "train/completion_loss", "validation/completion_loss",
        "learning_rate", "gradient_norm", "tokens_seen", "examples_seen", "tokens_per_second",
        "supervised_tokens_seen", "estimated_flops", "wall_seconds", "peak_gpu_memory_bytes",
    ]
    metrics_path = run_dir / "metrics.csv"
    mode = "a" if metrics_path.exists() and step else "w"
    handle = metrics_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if mode == "w":
        writer.writeheader()
    epoch_order: list[int] = []
    start_time = time.monotonic()
    elapsed = wall_seconds_before
    accumulated_batches = 0
    running_loss = 0.0
    running_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        while (
            epoch < max_epochs
            and step < max_steps
            and (target_flops is None or 6 * parameters * tokens_seen < target_flops)
        ):
            generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + epoch)
            shuffled = torch.randperm(len(train_indices), generator=generator).tolist()
            epoch_order = [train_indices[index] for index in shuffled]
            while (
                cursor < len(epoch_order)
                and step < max_steps
                and (target_flops is None or 6 * parameters * tokens_seen < target_flops)
            ):
                batch_indices = epoch_order[cursor : cursor + batch_size]
                loss_sum, target_tokens, input_tokens = _batch_loss(
                    model,
                    train_split,
                    batch_indices,
                    pad_id=pad_id,
                    device=device,
                    backward=True,
                )
                cursor += len(batch_indices)
                accumulated_batches += 1
                tokens_seen += input_tokens
                supervised_tokens_seen += target_tokens
                examples_seen += len(batch_indices)
                running_loss += loss_sum
                running_tokens += target_tokens
                if accumulated_batches < accumulation and cursor < len(epoch_order):
                    continue

                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(max(running_tokens, 1))
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.trainer.grad_clip))
                    .detach()
                    .item()
                )
                learning_rate = _cosine_lr(
                    step,
                    warmup_steps=warmup_steps,
                    total_steps=total_updates,
                    base_lr=base_lr,
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accumulated_batches = 0
                train_loss = running_loss / max(running_tokens, 1)
                running_loss = 0.0
                running_tokens = 0
                elapsed = max(wall_seconds_before + time.monotonic() - start_time, 1e-9)
                estimated_flops = 6 * parameters * tokens_seen
                peak_memory_seen = max(
                    peak_memory_seen,
                    int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                )
                values = {
                    "optimizer_step": step,
                    "epoch": epoch + cursor / len(epoch_order),
                    "train/completion_loss": train_loss,
                    "validation/completion_loss": "",
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "tokens_seen": tokens_seen,
                    "supervised_tokens_seen": supervised_tokens_seen,
                    "examples_seen": examples_seen,
                    "tokens_per_second": tokens_seen / elapsed,
                    "estimated_flops": estimated_flops,
                    "wall_seconds": elapsed,
                    "peak_gpu_memory_bytes": peak_memory_seen,
                }
                if run is not None:
                    run.log({key: value for key, value in values.items() if value != ""}, step=step)
                target_reached = target_flops is not None and 6 * parameters * tokens_seen >= target_flops
                if step % eval_every == 0 or cursor == len(epoch_order) or target_reached:
                    validation_loss = _evaluate_loss(
                        model,
                        validation_split,
                        pad_id=pad_id,
                        device=device,
                        batch_size=int(config.eval.batch_size),
                        max_examples=validation_cap,
                    )
                    values["validation/completion_loss"] = validation_loss
                    if validation_loss < best_val:
                        best_val = validation_loss
                        best_dir = run_dir / "best"
                        model.save_pretrained(best_dir, safe_serialization=True)
                        tokenizer = load_tokenizer(config.model)
                        tokenizer.save_pretrained(best_dir)
                        identity = {
                            "training_arm": trainer_name,
                            "training_phase": experiment_phase,
                            "model_revision": str(config.model.revision),
                            "model_config_sha256": cache_manifest["model_config_sha256"],
                            "checkpoint_config_sha256": sha256_file(best_dir / "config.json"),
                            "tokenizer_sha256": cache_manifest["tokenizer_sha256"],
                            "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                            "parameters": parameters,
                            "selected_step": step,
                            "validation_loss": validation_loss,
                            "source_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                            "source_sft_checkpoint": (
                                str(init_checkpoint)
                                if trainer_name == "continued_sft" and init_checkpoint else None
                            ),
                        }
                        save_model_identity(best_dir / "experiment_identity.json", identity)
                    if run is not None:
                        run.log({"validation/completion_loss": validation_loss}, step=step)
                _write_metric(writer, handle, values)
                if step % save_every == 0:
                    save_checkpoint(
                        latest_path,
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            epoch=epoch,
                            cursor=cursor,
                            best_val=best_val,
                            epoch_best_before=epoch_best_before,
                            no_improvement_epochs=no_improvement_epochs,
                            tokens_seen=tokens_seen,
                            examples_seen=examples_seen,
                            supervised_tokens_seen=supervised_tokens_seen,
                            peak_gpu_memory_bytes=peak_memory_seen,
                            wall_seconds=elapsed,
                            config_sha256=config_hash,
                        ),
                    )
                if target_flops is not None and 6 * parameters * tokens_seen >= target_flops:
                    break
            if cursor >= len(epoch_order):
                epoch += 1
                cursor = 0
                if best_val == math.inf:
                    raise RuntimeError("SFT epoch ended without a validation checkpoint")
                # One full epoch without an improvement ends optimization.
                if best_val < epoch_best_before:
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                epoch_best_before = best_val
                if (
                    target_flops is None
                    and no_improvement_epochs >= int(config.trainer.early_stopping_patience_epochs)
                ):
                    break
                save_checkpoint(
                    latest_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        epoch=epoch,
                        cursor=cursor,
                        best_val=best_val,
                        epoch_best_before=epoch_best_before,
                        no_improvement_epochs=no_improvement_epochs,
                        tokens_seen=tokens_seen,
                        examples_seen=examples_seen,
                        supervised_tokens_seen=supervised_tokens_seen,
                        peak_gpu_memory_bytes=peak_memory_seen,
                        wall_seconds=elapsed,
                        config_sha256=config_hash,
                    ),
                )
        save_checkpoint(
            latest_path,
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                cursor=cursor,
                best_val=best_val,
                epoch_best_before=epoch_best_before,
                no_improvement_epochs=no_improvement_epochs,
                tokens_seen=tokens_seen,
                examples_seen=examples_seen,
                supervised_tokens_seen=supervised_tokens_seen,
                peak_gpu_memory_bytes=peak_memory_seen,
                wall_seconds=elapsed,
                config_sha256=config_hash,
            ),
        )
    finally:
        handle.close()

    best_dir = run_dir / "best"
    if not best_dir.is_dir():
        raise RuntimeError("SFT completed without a validation-selected best checkpoint")
    identity_path = best_dir / "experiment_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    weight_files = sorted(best_dir.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(best_dir.glob("pytorch_model*.bin"))
    if not weight_files:
        raise RuntimeError("Selected SFT checkpoint has no saved model weights")
    checkpoint_digest = hashlib.sha256()
    identity["model_weight_files"] = {}
    for path in weight_files:
        weight_hash = sha256_file(path)
        identity["model_weight_files"][path.name] = weight_hash
        checkpoint_digest.update(path.name.encode("utf-8"))
        checkpoint_digest.update(weight_hash.encode("ascii"))
    identity["checkpoint_sha256"] = checkpoint_digest.hexdigest()
    save_model_identity(identity_path, identity)
    final_dir = run_dir / "final"
    final_identity = {
        **identity,
        "training_arm": trainer_name,
        "training_phase": experiment_phase,
        "optimizer_steps": step,
        "epochs_completed": epoch + (cursor / len(train_indices) if cursor else 0),
        "tokens_seen": tokens_seen,
        "estimated_additional_flops": 6 * parameters * tokens_seen,
        "compute_budget_flops": target_flops,
        "planned_optimizer_steps": total_updates,
        "planned_input_tokens": planned_tokens,
        "planned_epochs": planned_epochs,
        "planned_flops": 6 * parameters * planned_tokens if planned_tokens is not None else None,
        "source_sft_checkpoint": (
            str(init_checkpoint)
            if trainer_name == "continued_sft" and init_checkpoint else None
        ),
    }
    model.save_pretrained(final_dir, safe_serialization=True)
    load_tokenizer(config.model).save_pretrained(final_dir)
    final_identity["checkpoint_config_sha256"] = sha256_file(final_dir / "config.json")
    final_weights = sorted(final_dir.glob("model*.safetensors"))
    if not final_weights:
        final_weights = sorted(final_dir.glob("pytorch_model*.bin"))
    if not final_weights:
        raise RuntimeError("Final SFT model has no serialized weights")
    final_digest = hashlib.sha256()
    final_identity["model_weight_files"] = {}
    for path in final_weights:
        value = sha256_file(path)
        final_identity["model_weight_files"][path.name] = value
        final_digest.update(path.name.encode("utf-8"))
        final_digest.update(value.encode("ascii"))
    final_identity["checkpoint_sha256"] = final_digest.hexdigest()
    save_model_identity(final_dir / "experiment_identity.json", final_identity)
    selected_checkpoint = final_dir if trainer_name == "continued_sft" else best_dir
    result = {
        "step": step,
        "epoch": epoch,
        "training_examples": len(train_indices),
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        "supervised_tokens_seen": supervised_tokens_seen,
        "wall_seconds": wall_seconds_before + time.monotonic() - start_time,
        "tokens_per_second": tokens_seen
        / max(wall_seconds_before + time.monotonic() - start_time, 1e-9),
        "peak_gpu_memory_bytes": peak_memory_seen,
        "parameters": parameters,
        "best_validation_loss": best_val,
        "compute_budget_flops": target_flops,
        "compute_overshoot_flops": (
            max(0, 6 * parameters * tokens_seen - target_flops) if target_flops is not None else None
        ),
        "compute_matching_error": (
            (6 * parameters * tokens_seen - target_flops) / target_flops
            if target_flops is not None else None
        ),
        "planned_optimizer_steps": total_updates,
        "planned_input_tokens": planned_tokens,
        "planned_epochs": planned_epochs,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": (
            final_identity["checkpoint_sha256"]
            if selected_checkpoint == final_dir else identity["checkpoint_sha256"]
        ),
        "best_checkpoint": str(best_dir),
        "final_checkpoint": str(final_dir),
        "latest_checkpoint": str(latest_path),
        "additional_estimated_flops": 6 * parameters * tokens_seen,
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
