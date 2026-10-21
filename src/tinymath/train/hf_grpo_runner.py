"""GRPO training loop for the native-token Experiment 16 Qwen policy."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from tinymath.checkpoint import capture_rng_state, restore_rng_state, save_checkpoint
from tinymath.data.hf_cached_dataset import HFTokenSplit
from tinymath.data.hf_reasoning import score_native_reasoning, sha256_file
from tinymath.generation import RolloutConfig
from tinymath.model.hf_qwen import load_model, load_tokenizer, save_model_identity
from tinymath.train.grpo import group_outcome_rates, group_relative_advantages
from tinymath.train.hf_generation import NativeRollout, generate_native_rollout_group
from tinymath.train.hf_grpo import PromptRollouts, chunked_grpo_backward
from tinymath.train.hf_sft import _find_cache

LOGGER = logging.getLogger(__name__)


def _finalize_model(
    model: Any,
    tokenizer: Any,
    destination: Path,
    identity: dict[str, Any],
) -> str:
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    files = sorted(destination.glob("model*.safetensors"))
    if not files:
        files = sorted(destination.glob("pytorch_model*.bin"))
    if not files:
        raise RuntimeError("Final GRPO model has no serialized weight files")
    digest = hashlib.sha256()
    identity["model_weight_files"] = {}
    for path in files:
        value = sha256_file(path)
        identity["model_weight_files"][path.name] = value
        digest.update(path.name.encode("utf-8"))
        digest.update(value.encode("ascii"))
    identity["checkpoint_sha256"] = digest.hexdigest()
    identity["checkpoint_config_sha256"] = sha256_file(destination / "config.json")
    save_model_identity(destination / "experiment_identity.json", identity)
    return digest.hexdigest()


def _checkpoint_payload(
    *,
    model: Any,
    optimizer: Any,
    update: int,
    tokens_seen: int,
    policy_tokens: int,
    executor_tokens: int,
    estimated_flops: int,
    prompt_examples: int,
    wall_seconds: float,
    reward_mean: float,
    loss: float,
    peak_gpu_memory_bytes: int,
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "format": "exp16_hf_grpo_v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "tokens_seen": tokens_seen,
        "policy_tokens": policy_tokens,
        "executor_tokens": executor_tokens,
        "estimated_flops": estimated_flops,
        "prompt_examples": prompt_examples,
        "wall_seconds": wall_seconds,
        "reward_mean": reward_mean,
        "loss": loss,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "config_sha256": config_sha256,
        "rng_state": capture_rng_state(),
    }

def _completion_text(tokenizer: Any, rollout: NativeRollout) -> str:
    ids = rollout.token_ids
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def train_hf_grpo(
    *,
    config: Any,
    bundle_dir: Path,
    init_checkpoint: Path | None,
    run_dir: Path,
    run: Any,
) -> dict[str, Any]:
    phase = str(config.experiment.phase)
    if init_checkpoint is None and phase not in {"pilot", "smoke"}:
        raise ValueError("Main Experiment 16 GRPO must initialize from selected SFT weights")
    if init_checkpoint is not None and not init_checkpoint.is_dir():
        raise FileNotFoundError(f"Hugging Face SFT checkpoint directory not found: {init_checkpoint}")
    cache_dir, cache_manifest = _find_cache(config)
    if sha256_file(bundle_dir / "manifest.json") != cache_manifest["source_bundle_manifest_sha256"]:
        raise ValueError("GRPO bundle manifest does not match the native token cache source")
    rlvr_split = HFTokenSplit(cache_dir, "rlvr")
    prompts = list(rlvr_split.rows())
    if any(row.get("is_sub_task", False) for row in prompts):
        raise ValueError("Native GRPO prompt pool must contain only fixed parent rows")
    tokenizer = load_tokenizer(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Experiment 16 GRPO requires the admitted RTX 4090 with bf16 support")
    model = load_model(config.model, checkpoint=init_checkpoint, training=True).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    selected_identity = None
    if init_checkpoint is not None:
        selected_identity = json.loads(
            (init_checkpoint / "experiment_identity.json").read_text(encoding="utf-8")
        )
        if selected_identity.get("tokenizer_sha256") != cache_manifest["tokenizer_sha256"]:
            raise ValueError("Selected SFT checkpoint has a different native tokenizer")
        if selected_identity.get("model_revision") != str(config.model.revision):
            raise ValueError("Selected SFT checkpoint does not descend from the pinned Base revision")
    if hasattr(run, "config"):
        run.config.update(
            {
                "training_arm": "grpo",
                "model_parameters": parameters,
                "model_revision": str(config.model.revision),
                "tokenizer_sha256": cache_manifest["tokenizer_sha256"],
                "model_config_sha256": cache_manifest["model_config_sha256"],
                "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                "source_sft_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "source_sft_checkpoint_sha256": (
                    selected_identity.get("checkpoint_sha256") if selected_identity else None
                ),
            },
            allow_val_change=True,
        )

    rollout_config = RolloutConfig.from_config(config.rollout)
    if rollout_config.executor != "exact":
        raise ValueError("Experiment 16 requires the reasoning-v3 exact executor")
    updates = int(config.rlvr.updates)
    prompts_per_update = int(config.rlvr.prompts_per_batch)
    group_size = int(config.rlvr.group_size)
    temperature = float(config.rlvr.temperature)
    cap = int(config.rlvr.max_completion_tokens)
    clip_epsilon = float(config.rlvr.clip_epsilon)
    chunk_size = int(config.rlvr.rollout_chunk_size)
    if updates < 1 or prompts_per_update != 16 or group_size != 16 or chunk_size < 1:
        raise ValueError("Experiment 16 requires positive updates and 16 prompts by 16 completions")
    learning_rate = float(config.rlvr.learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0, foreach=False
    )
    config_identity = {
        "experiment": dict(config.experiment),
        "model": dict(config.model),
        "rlvr": dict(config.rlvr),
        "rollout": dict(config.rollout),
        "seed": int(config.seed),
        "source_sft_checkpoint_sha256": (
            selected_identity.get("checkpoint_sha256") if selected_identity else None
        ),
        "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
    }
    config_sha256 = hashlib.sha256(
        json.dumps(config_identity, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
    latest_path = run_dir / "latest.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_update = 0
    tokens_seen = policy_tokens_seen = executor_tokens_seen = estimated_flops_seen = 0
    prompt_examples = 0
    wall_seconds_before = 0.0
    peak_memory_seen = 0
    reward_mean = 0.0
    last_loss = 0.0
    if latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state.get("format") != "exp16_hf_grpo_v1" or state.get("config_sha256") != config_sha256:
            raise ValueError("Experiment 16 GRPO checkpoint identity/configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"])
        tokens_seen = int(state["tokens_seen"])
        policy_tokens_seen = int(state["policy_tokens"])
        executor_tokens_seen = int(state["executor_tokens"])
        estimated_flops_seen = int(state["estimated_flops"])
        prompt_examples = int(state["prompt_examples"])
        wall_seconds_before = float(state["wall_seconds"])
        reward_mean = float(state.get("reward_mean", 0.0))
        last_loss = float(state.get("loss", 0.0))
        peak_memory_seen = int(state.get("peak_gpu_memory_bytes", 0))
        restore_rng_state(state["rng_state"])

    fields = [
        "update", "reward/mean", "reward/format_component", "reward/success_component",
        "reward/answer_accuracy", "reward/full_success_rate", "reward/format_valid_rate",
        "reward/trace_valid_rate", "reward/capped_rate", "reward/malformed_rate",
        "group/mixed_rate", "group/constant_rate", "group/all_correct_rate",
        "group/all_wrong_rate", "group/all_format_only_rate", "policy_tokens",
        "executor_tokens", "generated_tokens", "mean_completion_tokens", "calls/count",
        "calls/correctness", "calls/failure_count", "loss", "policy_loss", "kl",
        "entropy", "clip_fraction", "grad_norm", "tokens_seen", "estimated_flops",
        "tokens_per_second", "wall_seconds", "peak_gpu_memory_bytes",
    ]
    metrics_path = run_dir / "metrics.csv"
    mode = "a" if metrics_path.exists() and start_update else "w"
    metrics_file = metrics_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(metrics_file, fieldnames=fields)
    if mode == "w":
        writer.writeheader()
    started = time.monotonic()
    save_every = int(config.rlvr.save_every)
    try:
        for update in range(start_update + 1, updates + 1):
            torch.cuda.reset_peak_memory_stats(device)
            chosen = random.sample(prompts, k=prompts_per_update)
            groups: list[PromptRollouts] = []
            scores_by_group = []
            lengths: list[int] = []
            for row in chosen:
                prompt_ids = tokenizer.encode(
                    f"Q: {row['question']}\n", add_special_tokens=False
                )
                if len(prompt_ids) >= int(config.model.context_length):
                    raise ValueError("Prepared GRPO prompt leaves no native-token context")
                rollouts = generate_native_rollout_group(
                    model,
                    tokenizer,
                    prompt_ids,
                    row,
                    group_size=group_size,
                    context_length=int(config.model.context_length),
                    max_completion_tokens=cap,
                    temperature=temperature,
                    rollout=rollout_config,
                )
                completions = []
                scores = []
                for rollout in rollouts:
                    text = _completion_text(tokenizer, rollout)
                    capped = rollout.termination_reason == "budget_exceeded"
                    failure = (
                        rollout.termination_reason
                        if rollout.termination_reason not in {"eos", "budget_exceeded"}
                        else None
                    )
                    score = score_native_reasoning(
                        text,
                        row,
                        capped=capped,
                        failure=failure,
                        **rollout_config.scoring_kwargs(),
                    )
                    completions.append(rollout)
                    scores.append(score)
                    lengths.append(len(rollout.token_ids))
                groups.append(PromptRollouts(prompt_ids=list(prompt_ids), completions=tuple(completions)))
                scores_by_group.append(scores)
            LOGGER.info("GRPO update %d: collected all %d sampled completions", update,
                        prompts_per_update * group_size)

            rewards = torch.tensor(
                [[score.reward for score in group] for group in scores_by_group],
                dtype=torch.float32,
                device=device,
            )
            advantages = group_relative_advantages(rewards)
            outcomes = group_outcome_rates(rewards)
            model.train()
            optimization = chunked_grpo_backward(
                model,
                groups,
                advantages,
                temperature=temperature,
                clip_epsilon=clip_epsilon,
                chunk_size=chunk_size,
                vocab_size=len(tokenizer),
            )
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.trainer.grad_clip))
                .detach()
                .item()
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            policy_tokens = sum(
                rollout.policy_tokens for group in groups for rollout in group.completions
            )
            executor_tokens = sum(
                rollout.executor_tokens for group in groups for rollout in group.completions
            )
            generated_tokens = policy_tokens + executor_tokens
            prompt_tokens = sum(
                len(group.prompt_ids) * len(group.completions) for group in groups
            )
            score_input_tokens = prompt_tokens + generated_tokens
            tokens_seen += score_input_tokens
            policy_tokens_seen += policy_tokens
            executor_tokens_seen += executor_tokens
            prompt_examples += prompts_per_update
            update_flops = 2 * parameters * score_input_tokens + 6 * parameters * score_input_tokens
            estimated_flops_seen += update_flops
            total_rewards = [score for group in scores_by_group for score in group]
            reward_mean = sum(score.reward for score in total_rewards) / len(total_rewards)
            calls = [score.call_count for score in total_rewards]
            calls_count = sum(calls)
            correct_calls = sum(score.correct_calls for score in total_rewards)
            reasons = [rollout.termination_reason for group in groups for rollout in group.completions]
            failure_counts = Counter(
                reason for reason in reasons if reason not in {"eos", "budget_exceeded"}
            )
            malformed_count = sum(value for key, value in failure_counts.items() if key != "disallowed_call")
            capped_count = sum(reason == "budget_exceeded" for reason in reasons)
            format_valid_count = sum(score.format_valid for score in total_rewards)
            full_success_count = sum(score.full_success for score in total_rewards)
            trace_valid_count = sum(score.trace_valid for score in total_rewards)
            answer_correct_count = sum(score.answer_correct for score in total_rewards)
            format_component = sum(score.format_reward for score in total_rewards) / len(total_rewards)
            success_component = sum(score.success_reward for score in total_rewards) / len(total_rewards)
            constant_rate = 1.0 - float(outcomes["mixed_group_rate"].item())
            elapsed = max(wall_seconds_before + time.monotonic() - started, 1e-9)
            last_loss = optimization["loss"]
            values = {
                "update": update,
                "reward/mean": reward_mean,
                "reward/format_component": format_component,
                "reward/success_component": success_component,
                "reward/answer_accuracy": answer_correct_count / len(total_rewards),
                "reward/full_success_rate": full_success_count / len(total_rewards),
                "reward/format_valid_rate": format_valid_count / len(total_rewards),
                "reward/trace_valid_rate": trace_valid_count / len(total_rewards),
                "reward/capped_rate": capped_count / len(total_rewards),
                "reward/malformed_rate": malformed_count / len(total_rewards),
                "group/mixed_rate": float(outcomes["mixed_group_rate"].item()),
                "group/constant_rate": constant_rate,
                "group/all_correct_rate": float(outcomes["all_correct_group_rate"].item()),
                "group/all_wrong_rate": float(outcomes["all_wrong_group_rate"].item()),
                "group/all_format_only_rate": float(outcomes["all_format_only_group_rate"].item()),
                "policy_tokens": policy_tokens,
                "executor_tokens": executor_tokens,
                "generated_tokens": generated_tokens,
                "mean_completion_tokens": sum(lengths) / len(lengths),
                "calls/count": calls_count,
                "calls/correctness": correct_calls / calls_count if calls_count else 0.0,
                "calls/failure_count": sum(failure_counts.values()),
                "loss": optimization["loss"],
                "policy_loss": optimization["policy_loss"],
                "kl": optimization["kl"],
                "entropy": optimization["entropy"],
                "clip_fraction": optimization["clip_fraction"],
                "grad_norm": grad_norm,
                "tokens_seen": tokens_seen,
                "estimated_flops": estimated_flops_seen,
                "tokens_per_second": tokens_seen / elapsed,
                "wall_seconds": elapsed,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            peak_memory_seen = max(peak_memory_seen, int(values["peak_gpu_memory_bytes"]))
            writer.writerow(values)
            metrics_file.flush()
            if run is not None:
                run.log(values, step=update)
            if update % save_every == 0 or update == updates:
                save_checkpoint(
                    latest_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        update=update,
                        tokens_seen=tokens_seen,
                        policy_tokens=policy_tokens_seen,
                        executor_tokens=executor_tokens_seen,
                        estimated_flops=estimated_flops_seen,
                        prompt_examples=prompt_examples,
                        wall_seconds=elapsed,
                        reward_mean=reward_mean,
                        loss=last_loss,
                        peak_gpu_memory_bytes=peak_memory_seen,
                        config_sha256=config_sha256,
                    ),
                )
    finally:
        metrics_file.close()

    final_dir = run_dir / "final"
    identity = {
        "training_arm": "grpo",
        "model_revision": str(config.model.revision),
        "model_config_sha256": cache_manifest["model_config_sha256"],
        "tokenizer_sha256": cache_manifest["tokenizer_sha256"],
        "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "parameters": parameters,
        "update": updates,
        "source_sft_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "source_sft_checkpoint_sha256": (
            selected_identity.get("checkpoint_sha256") if selected_identity else None
        ),
        "estimated_flops": estimated_flops_seen,
    }
    checkpoint_sha256 = _finalize_model(model, tokenizer, final_dir, identity)
    return {
        "update": updates,
        "prompt_examples": prompt_examples,
        "examples_drawn": prompt_examples,
        "tokens_seen": tokens_seen,
        "policy_tokens": policy_tokens_seen,
        "executor_tokens": executor_tokens_seen,
        "estimated_flops": estimated_flops_seen,
        "final_reward": reward_mean,
        "final_loss": last_loss,
        "wall_seconds": wall_seconds_before + time.monotonic() - started,
        "tokens_per_second": tokens_seen
        / max(wall_seconds_before + time.monotonic() - started, 1e-9),
        "peak_gpu_memory_bytes": peak_memory_seen,
        "checkpoint": str(final_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "latest_checkpoint": str(latest_path),
    }
