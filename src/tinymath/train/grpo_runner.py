"""End-to-end GRPO updates over the fixed disjoint RLVR prompt pool."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from tinymath.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.eval.qualitative import build_qualitative_table
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.rewards.verify import verify_answer
from tinymath.train.grpo import (
    completion_log_probs,
    group_outcome_rates,
    group_relative_advantages,
    grpo_objective,
    sample_group_rollouts,
)
from tinymath.train.metrics import truncate_csv_after_step


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_grpo(
    *,
    config: Any,
    bundle_dir: Path,
    init_checkpoint: Path,
    run_dir: Path,
    run: Any,
) -> dict[str, Any]:
    if not init_checkpoint.is_file():
        raise FileNotFoundError(f"SFT initialization checkpoint not found: {init_checkpoint}")
    tokenizer = CharacterTokenizer.load(bundle_dir / "char_tokenizer.json")
    prompts = _load_rows(bundle_dir / "rlvr.jsonl")
    if not prompts:
        raise ValueError("RLVR prompt pool is empty")
    initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("tokenizer_sha256") != tokenizer.sha256:
        raise ValueError("SFT checkpoint tokenizer does not match the RLVR data bundle")
    model_config = GPTConfig.from_mapping(initial["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMathGPT(model_config).to(device)
    model.load_state_dict(initial["model"])
    use_bf16 = bool(config.trainer.bf16 and device.type == "cuda")
    reference: TinyMathGPT | None = None
    if float(config.rlvr.kl_coefficient) > 0:
        reference = TinyMathGPT(model_config).to(device)
        reference.load_state_dict(initial["model"])
        reference.eval()
        reference.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.rlvr.learning_rate),
        weight_decay=float(config.trainer.weight_decay),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / "latest.pt"
    start_update = 0
    tokens_seen = 0
    estimated_flops_seen = 0
    parameter_count = model.count_parameters()
    if latest_path.exists():
        resumed = torch.load(latest_path, map_location="cpu", weights_only=False)
        if resumed.get("tokenizer_sha256") != tokenizer.sha256:
            raise ValueError("GRPO checkpoint tokenizer does not match the RLVR data bundle")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_update = int(resumed["update"])
        tokens_seen = int(resumed.get("tokens_seen", 0))
        estimated_flops_seen = int(
            resumed.get("estimated_flops", 6 * parameter_count * tokens_seen)
        )
        restore_rng_state(resumed["rng_state"])

    updates = int(config.rlvr.updates)
    qualitative_every = max(1, int(getattr(getattr(config, "qualitative", None), "every_steps", 20_000)))
    qualitative_count = 0
    define_metric = getattr(run, "define_metric", None)
    if callable(define_metric):
        define_metric("optimizer_step")
        define_metric("grpo/update")
        define_metric("grpo/*", step_metric="grpo/update")
        define_metric("qualitative/*", step_metric="optimizer_step")
    prompts_per_batch = int(config.rlvr.prompts_per_batch)
    group_size = int(config.rlvr.group_size)
    temperature = float(config.rlvr.temperature)
    max_completion = int(config.rlvr.max_completion_tokens)
    clip_epsilon = float(config.rlvr.clip_epsilon)
    kl_coefficient = float(config.rlvr.kl_coefficient)
    if prompts_per_batch < 1 or group_size < 2 or updates < 1:
        raise ValueError("GRPO requires positive updates/prompts and group_size >= 2")
    metrics_path = run_dir / "metrics.csv"
    if start_update:
        truncate_csv_after_step(metrics_path, step_column="update", step=start_update)
    mode = "a" if metrics_path.exists() and start_update else "w"
    metrics_file = metrics_path.open(mode, newline="", encoding="utf-8")
    fields = [
        "update", "reward/mean", "reward/format_valid_rate", "reward/verifier_failure_rate",
        "group/mixed_rate", "group/all_correct_rate", "group/all_wrong_rate", "loss",
        "policy_loss", "kl", "entropy", "clip_fraction", "grad_norm", "tokens_seen", "wall_seconds",
        "tokens_per_second", "estimated_flops", "peak_gpu_memory_bytes",
        "flops_estimator_version",
    ]
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    if mode == "w":
        writer.writeheader()
    started = time.monotonic()
    try:
        for update in range(start_update + 1, updates + 1):
            chosen = random.sample(prompts, k=min(prompts_per_batch, len(prompts)))
            optimizer.zero_grad(set_to_none=True)
            metric_groups: list[dict[str, torch.Tensor]] = []
            batch_correct = batch_valid = batch_total = token_total = 0
            optimized_groups = 0
            policy_losses: list[float] = []
            kl_values: list[float] = []
            entropy_values: list[float] = []
            clip_values: list[float] = []
            estimated_flops_update = 0
            for row in chosen:
                prompt_text = format_prompt(row["question"])
                prompt_ids = torch.tensor(
                    [tokenizer.encode(prompt_text, add_bos=True)], dtype=torch.long, device=device
                )
                room = model_config.context_length - prompt_ids.shape[1]
                if room < 1:
                    raise ValueError("Prepared RLVR prompt leaves no room for a completion")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    rollouts = sample_group_rollouts(
                        model,
                        prompt_ids,
                        group_size=group_size,
                        max_completion_tokens=min(max_completion, room),
                        temperature=temperature,
                        eos_id=tokenizer.eos_id,
                        pad_id=tokenizer.pad_id,
                    )
                completions = rollouts.completion_ids[0]
                decoded = [
                    tokenizer.decode(
                        completion[: int(mask.sum().item())].tolist(), skip_special=True
                    )
                    for completion, mask in zip(completions, rollouts.completion_mask[0], strict=True)
                ]
                outcome_pairs = [
                    verify_answer(text, row["answer"], question=row["question"])
                    for text in decoded
                ]
                rewards = torch.tensor(
                    [int(correct) for correct, _valid in outcome_pairs], dtype=torch.float32, device=device
                )
                valid_count = sum(int(valid) for _correct, valid in outcome_pairs)
                batch_correct += int(rewards.sum().item())
                batch_valid += valid_count
                batch_total += group_size
                token_total += int(rollouts.completion_mask.sum().item())
                advantages = group_relative_advantages(rewards.unsqueeze(0))
                metric_groups.append(group_outcome_rates(rewards.unsqueeze(0)))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    current = completion_log_probs(
                        model, prompt_ids, rollouts.completion_ids, temperature=temperature
                    )
                    reference_logs = (
                        completion_log_probs(reference, prompt_ids, rollouts.completion_ids,
                                             temperature=temperature)
                        if reference is not None else None
                    )
                metrics = grpo_objective(
                    current,
                    rollouts.old_log_probs,
                    advantages,
                    rollouts.completion_mask,
                    clip_epsilon=clip_epsilon,
                    reference_log_probs=reference_logs,
                    kl_coefficient=kl_coefficient,
                    entropy=rollouts.entropy,
                )
                if torch.any(advantages):
                    (metrics.loss / len(chosen)).backward()
                    optimized_groups += 1
                completion_length = completions.shape[-1]
                rollout_positions = group_size * (
                    completion_length * prompt_ids.shape[1]
                    + completion_length * (completion_length - 1) // 2
                )
                policy_positions = group_size * (prompt_ids.shape[1] + completion_length)
                policy_factor = 6 if torch.any(advantages) else 2
                reference_factor = 2 if reference is not None else 0
                estimated_flops_update += parameter_count * (
                    2 * rollout_positions
                    + (policy_factor + reference_factor) * policy_positions
                )
                policy_losses.append(float(metrics.policy_loss.detach().item()))
                kl_values.append(float(metrics.kl.detach().item()))
                entropy_values.append(float(metrics.entropy.detach().item()))
                clip_values.append(float(metrics.clip_fraction.detach().item()))
            grad_norm_value = 0.0
            if optimized_groups:
                grad_norm_value = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.trainer.grad_clip)).item()
                )
                optimizer.step()
            tokens_seen += token_total
            estimated_flops_seen += estimated_flops_update
            reward_mean = batch_correct / batch_total

            def mean(values: list[float]) -> float:
                return sum(values) / max(1, len(values))
            mixed = mean([item["mixed_group_rate"].item() for item in metric_groups])
            all_correct = mean([item["all_correct_group_rate"].item() for item in metric_groups])
            all_wrong = mean([item["all_wrong_group_rate"].item() for item in metric_groups])
            values = {
                "update": update,
                "reward/mean": reward_mean,
                "reward/format_valid_rate": batch_valid / batch_total,
                "reward/verifier_failure_rate": 1 - batch_valid / batch_total,
                "group/mixed_rate": mixed,
                "group/all_correct_rate": all_correct,
                "group/all_wrong_rate": all_wrong,
                "loss": mean(policy_losses) + kl_coefficient * mean(kl_values),
                "policy_loss": mean(policy_losses),
                "kl": mean(kl_values),
                "entropy": mean(entropy_values),
                "clip_fraction": mean(clip_values),
                "grad_norm": grad_norm_value,
                "tokens_seen": tokens_seen,
                "wall_seconds": time.monotonic() - started,
                "tokens_per_second": tokens_seen / max(1e-9, time.monotonic() - started),
                "estimated_flops": estimated_flops_seen,
                "flops_estimator_version": "dense_position_v2",
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
            writer.writerow(values)
            metrics_file.flush()
            if update % int(config.trainer.save_every) == 0 or update == updates:
                save_checkpoint(
                    latest_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "tokens_seen": tokens_seen,
                        "estimated_flops": estimated_flops_seen,
                        "flops_estimator_version": "dense_position_v2",
                        "rng_state": capture_rng_state(),
                        "tokenizer_sha256": tokenizer.sha256,
                        "model_config": model_config.__dict__,
                        "training_arm": "grpo",
                        "source_sft_checkpoint": str(init_checkpoint),
                    },
                )
                save_checkpoint(
                    run_dir / "latest_inference.pt",
                    {
                        "model": model.state_dict(),
                        "update": update,
                        "tokenizer_sha256": tokenizer.sha256,
                        "model_config": model_config.__dict__,
                        "training_arm": "grpo",
                    },
                )
            qualitative_due = update % qualitative_every == 0 or update == updates
            if qualitative_due:
                qualitative_table, qualitative_count = build_qualitative_table(
                    model=model,
                    tokenizer=tokenizer,
                    split_rows={
                        "train": prompts,
                        "validation": _load_rows(bundle_dir / "sft_validation.jsonl"),
                        "test": _load_rows(bundle_dir / "test.jsonl"),
                    },
                    training_arm="grpo",
                    seed=int(getattr(config, "seed", 0)),
                    max_new_tokens=int(
                        getattr(getattr(config, "eval", None), "max_new_tokens", max_completion)
                    ),
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
            if (
                update % int(config.tracking.log_every) == 0
                or update == 1
                or update == updates
                or qualitative_due
            ):
                logged = {"optimizer_step": update}
                for key, value in values.items():
                    if key == "update":
                        logged["grpo/update"] = value
                    elif key == "qualitative/examples":
                        logged[key] = value
                    else:
                        logged[f"grpo/{key}"] = value
                run.log(logged)
    finally:
        metrics_file.close()
    return {
        "update": updates,
        "tokens_seen": tokens_seen,
        "estimated_flops": estimated_flops_seen,
        "checkpoint": str(latest_path),
        "inference_checkpoint": str(run_dir / "latest_inference.pt"),
        "metrics_path": str(metrics_path),
        "final_reward": reward_mean,
        "qualitative_examples": qualitative_count,
    }
