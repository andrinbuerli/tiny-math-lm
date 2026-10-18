"""End-to-end GRPO updates over the fixed disjoint RLVR prompt pool."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
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
from tinymath.data.completions import parse_reasoning_completion
from tinymath.data.prompts import format_prompt
from tinymath.data.trace_explanations import transcribe_scratchpad
from tinymath.eval.qualitative import build_qualitative_table
from tinymath.generation import RolloutConfig
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.rewards.reasoning import score_v3
from tinymath.rewards.verify import verify_answer
from tinymath.train.budgets import budget_rows
from tinymath.train.grpo import (
    completion_log_probs,
    group_outcome_rates,
    group_relative_advantages,
    grpo_objective,
    sample_group_rollouts,
)
from tinymath.train.metrics import elapsed_seconds_through_step, truncate_csv_after_step

CALL_FAILURES = (
    "disallowed_call",
    "malformed_request", "scalar_limit", "noncanonical_operand", "division_by_zero",
    "incorrect_call_result", "call_limit", "unencodable_result",
)

def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_completion(
    token_ids: list[int], tokenizer: CharacterTokenizer, row: dict[str, str]
) -> tuple[float, bool, str | None, bool | None, bool | None]:
    """Return reward, answer validity, trace, reference match, and trace validity."""
    if tokenizer.protocol == "reasoning_v3":
        score = score_v3(token_ids, tokenizer, row)
        return score.reward, score.format_valid, score.scratchpad, score.reference_match, score.trace_valid
    if tokenizer.protocol in {"reasoning_v1", "reasoning_v2"}:
        try:
            parsed = parse_reasoning_completion(token_ids, tokenizer)
        except ValueError:
            return False, False, None, False, False
        try:
            trace_valid = bool(
                transcribe_scratchpad(parsed.scratchpad, version=tokenizer.protocol)
            )
        except ValueError:
            trace_valid = False
        correct, valid = verify_answer(parsed.answer, row["answer"], question=row["question"])
        reference_match = parsed.scratchpad == row.get("scratchpad")
        return correct and valid and trace_valid, valid, parsed.scratchpad, reference_match, trace_valid
    text = tokenizer.decode(token_ids, skip_special=True).strip()
    correct, valid = verify_answer(text, row["answer"], question=row["question"])
    return correct and valid, valid, None, None, None


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
    rollout_config = RolloutConfig.from_config(getattr(config, "rollout", None))
    rollout_identity = rollout_config.identity()
    if rollout_config.executor == "exact" and tokenizer.protocol != "reasoning_v3":
        raise ValueError("Exact call execution requires reasoning_v3")
    if tokenizer.protocol == "reasoning_v3" and hasattr(run, "config"):
        run.config.update({"rollout": rollout_identity}, allow_val_change=True)
    prompts = [row for row in _load_rows(bundle_dir / "rlvr.jsonl") if not row.get("is_sub_task", False)]
    experiment = getattr(config, "experiment", {})
    if (
        int(experiment.get("number", 0) or 0) == 11
        and str(experiment.get("phase", "main")) in {"smoke", "pilot"}
    ):
        prompts = budget_rows(
            prompts,
            examples=int(experiment.get("data_budget", 0)),
            seed=int(config.seed),
        )
    if not prompts:
        raise ValueError("RLVR prompt pool is empty")
    initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    if initial.get("tokenizer_sha256") != tokenizer.sha256:
        raise ValueError("SFT checkpoint tokenizer does not match the RLVR data bundle")
    model_config = GPTConfig.from_mapping(initial["model_config"])
    manifest_path = bundle_dir / "manifest.json"
    compatibility = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("context_length", model_config.context_length)) != model_config.context_length:
            raise ValueError("Model context length does not match the RLVR data bundle")
        compatibility = compatibility_metadata(
            protocol=tokenizer.protocol,
            annotation_sha256=(manifest.get("annotation") or {}).get("annotation_sha256"),
            tokenizer_sha256=tokenizer.sha256,
            bundle_sha256=sha256_file(manifest_path),
            context_length=model_config.context_length,
        )
    elif tokenizer.protocol != "answer_only":
        raise ValueError("Reasoning GRPO requires a provenance-bearing annotated bundle")
    if compatibility is not None:
        validate_checkpoint_compatibility(
            initial,
            compatibility,
            allow_legacy_answer_only=(tokenizer.protocol == "answer_only"),
        )
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
    examples_drawn = 0
    policy_tokens_seen = executor_tokens_seen = 0
    reward_mean = 0.0
    nonpadding_input_tokens = 0
    padded_compute_positions = 0
    estimated_flops_seen = 0
    parameter_count = model.count_parameters()
    if hasattr(run, "config"):
        run.config.update({"model_parameters": parameter_count}, allow_val_change=True)
    if latest_path.exists():
        resumed = torch.load(latest_path, map_location="cpu", weights_only=False)
        if resumed.get("tokenizer_sha256") != tokenizer.sha256:
            raise ValueError("GRPO checkpoint tokenizer does not match the RLVR data bundle")
        if compatibility is not None:
            validate_checkpoint_compatibility(
                resumed,
                compatibility,
                allow_legacy_answer_only=(tokenizer.protocol == "answer_only"),
            )
        if tokenizer.protocol == "reasoning_v3" and resumed.get("rollout") != rollout_identity:
            raise ValueError("GRPO resume rollout/reward settings do not match checkpoint")
        policy_tokens_seen = int(resumed.get("policy_tokens_seen", resumed.get("tokens_seen", 0)))
        executor_tokens_seen = int(resumed.get("executor_tokens_seen", 0))
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_update = int(resumed["update"])
        reward_mean = float(resumed.get("last_reward_mean", 0.0))
        tokens_seen = int(resumed.get("tokens_seen", 0))
        examples_drawn = int(resumed.get("examples_drawn", 0))
        nonpadding_input_tokens = int(resumed.get("nonpadding_input_tokens", 0))
        padded_compute_positions = int(resumed.get("padded_compute_positions", 0))
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
        for metric in ("train/*", "val/*", "test/*"):
            define_metric(metric, step_metric="optimizer_step")
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
    elapsed_before = (
        elapsed_seconds_through_step(metrics_path, step_column="update", step=start_update)
        if start_update
        else 0.0
    )
    fields = [
        "update", "reward/mean", "reward/format_valid_rate", "reward/verifier_failure_rate",
        "reward/trace_valid_rate", "reward/trace_reference_exact_match_rate",
        "reward/capped_rate", "reward/malformed_rate",
        "group/mixed_rate", "group/all_correct_rate", "group/all_wrong_rate", "loss",
        "policy_loss", "kl", "entropy", "clip_fraction", "grad_norm", "tokens_seen",
        "policy_tokens", "executor_tokens", "transcript_tokens", "group/all_format_only_rate",
        "reward/format_component", "reward/success_component", "reward/full_success_rate",
        "reward/answer_accuracy", "reward/format_only_rate", "calls/per_completion",
        "calls/correctness", "calls/executor_failure_rate", "calls/no_call_success_rate",
        "calls/policy_violation_rate",
        "generated_tokens", "examples_drawn", "nonpadding_input_tokens",
        "padded_compute_positions", "wall_seconds",
        "tokens_per_second", "estimated_flops", "peak_gpu_memory_bytes",
        "flops_estimator_version", "train/greedy_accuracy", "val/greedy_accuracy",
        "test/greedy_accuracy",
    ] + [f"calls/failure/{reason}" for reason in CALL_FAILURES]
    if start_update:
        truncate_csv_after_step(
            metrics_path,
            step_column="update",
            step=start_update,
            fieldnames=fields,
        )
    mode = "a" if metrics_path.exists() and start_update else "w"
    metrics_file = metrics_path.open(mode, newline="", encoding="utf-8")
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
            reward_total = 0.0
            format_component = success_component = answer_correct = format_only = 0
            calls_total = correct_calls = no_call_count = no_call_success = 0
            policy_token_total = executor_token_total = 0
            failures: Counter[str] = Counter()
            batch_capped = batch_malformed = batch_trace_valid = batch_trace_reference_match = 0
            optimized_groups = 0
            policy_losses: list[float] = []
            kl_values: list[float] = []
            entropy_values: list[float] = []
            clip_values: list[float] = []
            estimated_flops_update = 0
            nonpadding_input_tokens_update = 0
            padded_compute_positions_update = 0
            for row in chosen:
                prompt_text = format_prompt(row["question"], protocol=tokenizer.protocol)
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
                        tokenizer=tokenizer, rollout=rollout_config,
                        task_rows=[row],
                    )
                completions = rollouts.completion_ids[0]
                generated_ids = [
                    completion[: int(mask.sum().item())].tolist()
                    for completion, mask in zip(completions, rollouts.completion_mask[0], strict=True)
                ]
                v3_scores = []
                if tokenizer.protocol == "reasoning_v3":
                    reasons = rollouts.termination_reasons or tuple(
                        "eos" if ids and ids[-1] == tokenizer.eos_id else "budget_exceeded"
                        for ids in generated_ids
                    )
                    for ids, reason in zip(generated_ids, reasons, strict=True):
                        failed = reason not in {"eos", "budget_exceeded"}
                        if failed:
                            failures[reason] += 1
                        score = score_v3(
                            ids, tokenizer, row, capped=reason == "budget_exceeded",
                            failure=reason if failed else None, **rollout_config.scoring_kwargs(),
                        )
                        v3_scores.append(score)
                        format_component += score.format_reward
                        success_component += score.success_reward
                        answer_correct += score.answer_correct
                        format_only += 0 < score.reward < 1
                        if not score.call_policy_valid and not failed:
                            failures["disallowed_call"] += 1
                        calls_total += score.call_count
                        correct_calls += score.correct_calls
                        no_call_count += score.format_valid and score.call_count == 0
                        no_call_success += score.full_success and score.call_count == 0
                    outcomes = [(v.reward, v.format_valid, v.scratchpad, v.reference_match, v.trace_valid)
                                for v in v3_scores]
                else:
                    outcomes = [_score_completion(ids, tokenizer, row) for ids in generated_ids]
                rewards = torch.tensor(
                    [float(correct) for correct, _valid, _trace, _reference_match, _trace_valid in outcomes],
                    dtype=torch.float32,
                    device=device,
                )
                valid_count = sum(int(valid) for _correct, valid, _trace, _match, _trace_valid in outcomes)
                batch_capped += sum(reason == "budget_exceeded" for reason in reasons) if v3_scores else sum(
                    int(len(ids) >= min(max_completion, room) and (not ids or ids[-1] != tokenizer.eos_id))
                    for ids in generated_ids
                )
                batch_malformed += group_size - valid_count
                batch_trace_valid += sum(
                    int(trace_valid) if trace_valid is not None else int(valid)
                    for _correct, valid, _trace, _match, trace_valid in outcomes
                )
                batch_trace_reference_match += sum(
                    int(reference_match is True)
                    for _correct, _valid, _trace, reference_match, _trace_valid in outcomes
                )
                reward_total += float(rewards.sum().item())
                batch_correct += int((rewards == 1).sum().item())
                policy_token_total += int(rollouts.loss_mask.sum().item())
                executor_token_total += int((rollouts.completion_mask & ~rollouts.loss_mask).sum().item())
                batch_valid += valid_count
                batch_total += group_size
                token_total += int(rollouts.completion_mask.sum().item())
                nonpadding_input_tokens_update += prompt_ids.numel() * group_size
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
                    rollouts.loss_mask,
                    clip_epsilon=clip_epsilon,
                    reference_log_probs=reference_logs,
                    kl_coefficient=kl_coefficient,
                    entropy=rollouts.entropy,
                )
                if torch.any(advantages):
                    (metrics.loss / len(chosen)).backward()
                    optimized_groups += 1
                completion_length = completions.shape[-1]
                # Rollout decoding uses a KV cache: the prompt is prefetched
                # once and each generated position is forwarded once. The
                # policy pass still evaluates the padded full completion batch.
                rollout_positions = group_size * (prompt_ids.shape[1] + completion_length)
                policy_positions = group_size * (prompt_ids.shape[1] + completion_length)
                padded_compute_positions_update += (
                    rollout_positions
                    + policy_positions * (1 + int(reference is not None))
                )
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
            policy_tokens_seen += policy_token_total
            executor_tokens_seen += executor_token_total
            examples_drawn += len(chosen)
            nonpadding_input_tokens += nonpadding_input_tokens_update
            padded_compute_positions += padded_compute_positions_update
            estimated_flops_seen += estimated_flops_update
            reward_mean = reward_total / batch_total

            def mean(values: list[float]) -> float:
                return sum(values) / max(1, len(values))
            mixed = mean([item["mixed_group_rate"].item() for item in metric_groups])
            all_correct = mean([item["all_correct_group_rate"].item() for item in metric_groups])
            all_wrong = mean([item["all_wrong_group_rate"].item() for item in metric_groups])
            elapsed = elapsed_before + time.monotonic() - started
            values = {
                "update": update,
                "reward/mean": reward_mean,
                **{f"calls/failure/{reason}": failures[reason] for reason in CALL_FAILURES},
                "reward/full_success_rate": batch_correct / batch_total,
                "reward/format_component": format_component / batch_total,
                "reward/success_component": success_component / batch_total,
                "reward/answer_accuracy": answer_correct / batch_total if v3_scores else batch_correct / batch_total,
                "reward/format_only_rate": format_only / batch_total,
                "group/all_format_only_rate": mean([item["all_format_only_group_rate"].item() for item in metric_groups]),
                "calls/per_completion": calls_total / batch_total,
                "calls/correctness": correct_calls / calls_total if calls_total else 0.0,
                "calls/executor_failure_rate": sum(failures.values()) / batch_total,
                "calls/policy_violation_rate": failures["disallowed_call"] / batch_total,
                "calls/no_call_success_rate": no_call_success / no_call_count if no_call_count else 0.0,
                "policy_tokens": policy_tokens_seen,
                "executor_tokens": executor_tokens_seen,
                "transcript_tokens": tokens_seen,
                "reward/format_valid_rate": batch_valid / batch_total,
                "reward/verifier_failure_rate": 1 - batch_valid / batch_total,
                "reward/trace_valid_rate": batch_trace_valid / batch_total,
                "reward/trace_reference_exact_match_rate": batch_trace_reference_match / batch_total,
                "reward/capped_rate": batch_capped / batch_total,
                "reward/malformed_rate": batch_malformed / batch_total,
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
                "examples_drawn": examples_drawn,
                "nonpadding_input_tokens": nonpadding_input_tokens,
                "generated_tokens": policy_tokens_seen,
                "padded_compute_positions": padded_compute_positions,
                "wall_seconds": elapsed,
                "tokens_per_second": tokens_seen / max(1e-9, elapsed),
                "estimated_flops": estimated_flops_seen,
                "flops_estimator_version": "kv_cached_position_v3",
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
            if update % int(config.trainer.save_every) == 0 or update == updates:
                save_checkpoint(
                    latest_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update,
                        "last_reward_mean": reward_mean,
                        "tokens_seen": tokens_seen,
                        "policy_tokens_seen": policy_tokens_seen,
                        "executor_tokens_seen": executor_tokens_seen,
                        **({"rollout": rollout_identity} if tokenizer.protocol == "reasoning_v3" else {}),
                        "examples_drawn": examples_drawn,
                        "nonpadding_input_tokens": nonpadding_input_tokens,
                        "padded_compute_positions": padded_compute_positions,
                        "estimated_flops": estimated_flops_seen,
                        "flops_estimator_version": "kv_cached_position_v3",
                        "rng_state": capture_rng_state(),
                        "tokenizer_sha256": tokenizer.sha256,
                        **({"compatibility": compatibility} if compatibility is not None else {}),
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
                        **({"rollout": rollout_identity} if tokenizer.protocol == "reasoning_v3" else {}),
                        "tokenizer_sha256": tokenizer.sha256,
                        **({"compatibility": compatibility} if compatibility is not None else {}),
                        "model_config": model_config.__dict__,
                        "training_arm": "grpo",
                        "source_sft_checkpoint": str(init_checkpoint.resolve()),
                    },
                )
            qualitative_due = update % qualitative_every == 0 or update == updates
            if qualitative_due:
                qualitative_table, qualitative_count, greedy_accuracy = build_qualitative_table(
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
                    rollout=rollout_config,
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
                for split, value in greedy_accuracy.items():
                    metric_name = "val/greedy_accuracy" if split == "validation" else f"{split}/greedy_accuracy"
                    values[metric_name] = value
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
                    elif key == "qualitative/examples" or key in {
                        "train/greedy_accuracy", "val/greedy_accuracy", "test/greedy_accuracy"
                    }:
                        logged[key] = value
                    else:
                        logged[f"grpo/{key}"] = value
                for reason, count in failures.items():
                    logged[f"grpo/calls/failure/{reason}"] = count
                run.log(logged)
            writer.writerow({key: value for key, value in values.items() if key in fields})
            metrics_file.flush()
    finally:
        metrics_file.close()
    return {
        "update": updates,
        "prompt_examples": len(prompts),
        "tokens_seen": tokens_seen,
        "policy_tokens": policy_tokens_seen,
        "executor_tokens": executor_tokens_seen,
        "generated_tokens": policy_tokens_seen,
        "examples_drawn": examples_drawn,
        "nonpadding_input_tokens": nonpadding_input_tokens,
        "padded_compute_positions": padded_compute_positions,
        "estimated_flops": estimated_flops_seen,
        "checkpoint": str(latest_path),
        "inference_checkpoint": str(run_dir / "latest_inference.pt"),
        "metrics_path": str(metrics_path),
        "final_reward": reward_mean,
        "qualitative_examples": qualitative_count,
    }
