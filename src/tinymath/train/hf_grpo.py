"""Native-token GRPO policy scoring with observation masks and bounded gradients."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from tinymath.train.grpo import grpo_objective
from tinymath.train.hf_generation import NativeRollout


@dataclass(frozen=True)
class PromptRollouts:
    prompt_ids: list[int]
    completions: tuple[NativeRollout, ...]


def native_policy_statistics(
    model: Any,
    prompt_ids: Sequence[int],
    rollout: NativeRollout,
    *,
    temperature: float,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute current token statistics; executor observations have zeroed values/masks."""
    if temperature <= 0:
        raise ValueError("GRPO scoring temperature must be positive")
    device = next(model.parameters()).device
    completion = rollout.token_ids
    if not completion:
        raise ValueError("Cannot score an empty generated completion")
    full = torch.tensor([[*prompt_ids, *completion]], dtype=torch.long, device=device)
    attention = torch.ones_like(full)
    outputs = model(input_ids=full, attention_mask=attention, use_cache=False)
    start = len(prompt_ids) - 1
    logits = outputs.logits[0, start : start + len(completion)].float() / temperature
    if vocab_size is not None:
        if logits.shape[-1] < vocab_size:
            raise ValueError("Model output vocabulary is smaller than the native tokenizer")
        if logits.shape[-1] > vocab_size:
            logits = logits.clone()
            logits[..., vocab_size:] = -torch.inf
    log_distribution = F.log_softmax(logits, dim=-1)
    target_ids = torch.tensor(completion, dtype=torch.long, device=device)
    current = log_distribution.gather(-1, target_ids[:, None]).squeeze(-1)
    probabilities = log_distribution.exp()
    finite_log_distribution = log_distribution.masked_fill(~torch.isfinite(log_distribution), 0.0)
    entropy = -(probabilities * finite_log_distribution).sum(dim=-1)
    policy_mask = torch.tensor(rollout.policy_mask, dtype=torch.bool, device=device)
    old = torch.tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    if not (len(completion) == len(policy_mask) == len(old)):
        raise ValueError("Rollout token, action mask, and old-log-probability lengths disagree")
    current = torch.where(policy_mask, current, torch.zeros_like(current))
    entropy = torch.where(policy_mask, entropy, torch.zeros_like(entropy))
    return current, old, entropy, policy_mask


def _batched_policy_statistics(
    model: Any,
    trajectories: Sequence[tuple[Sequence[int], NativeRollout]],
    *,
    temperature: float,
    vocab_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a bounded trajectory batch with right padding and observation masks."""
    if not trajectories:
        raise ValueError("At least one GRPO trajectory is required")
    if temperature <= 0:
        raise ValueError("GRPO scoring temperature must be positive")
    device = next(model.parameters()).device
    max_length = max(len(prompt) + len(rollout.token_ids) for prompt, rollout in trajectories)
    max_completion = max(len(rollout.token_ids) for _prompt, rollout in trajectories)
    if max_completion == 0:
        raise ValueError("Cannot score an empty generated completion")
    input_ids = torch.zeros((len(trajectories), max_length), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    target_ids = torch.zeros((len(trajectories), max_completion), dtype=torch.long, device=device)
    old_log_probs = torch.zeros((len(trajectories), max_completion), dtype=torch.float32, device=device)
    policy_mask = torch.zeros((len(trajectories), max_completion), dtype=torch.bool, device=device)
    for row_index, (prompt_ids, rollout) in enumerate(trajectories):
        prompt_length = len(prompt_ids)
        completion_length = len(rollout.token_ids)
        if not prompt_length or not completion_length:
            raise ValueError("Every scored trajectory needs a prompt and completion")
        if not (
            completion_length == len(rollout.policy_mask) == len(rollout.old_log_probs)
        ):
            raise ValueError("Rollout token, action mask, and old-log-probability lengths disagree")
        input_ids[row_index, :prompt_length] = torch.as_tensor(prompt_ids, device=device)
        input_ids[row_index, prompt_length : prompt_length + completion_length] = torch.as_tensor(
            rollout.token_ids, device=device
        )
        attention[row_index, : prompt_length + completion_length] = 1
        target_ids[row_index, :completion_length] = torch.as_tensor(rollout.token_ids, device=device)
        old_log_probs[row_index, :completion_length] = torch.as_tensor(
            rollout.old_log_probs, dtype=torch.float32, device=device
        )
        policy_mask[row_index, :completion_length] = torch.as_tensor(
            rollout.policy_mask, dtype=torch.bool, device=device
        )

    outputs = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    row_indices = torch.arange(len(trajectories), device=device)[:, None]
    completion_offsets = torch.arange(max_completion, device=device)[None, :]
    prompt_lengths = torch.tensor(
        [len(prompt) for prompt, _rollout in trajectories], dtype=torch.long, device=device
    )[:, None]
    logit_positions = prompt_lengths - 1 + completion_offsets
    logits = outputs.logits[row_indices, logit_positions].float() / temperature
    if vocab_size is not None:
        if logits.shape[-1] < vocab_size:
            raise ValueError("Model output vocabulary is smaller than the native tokenizer")
        if logits.shape[-1] > vocab_size:
            logits = logits.clone()
            logits[..., vocab_size:] = -torch.inf
    log_distribution = F.log_softmax(logits, dim=-1)
    current = log_distribution.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    probabilities = log_distribution.exp()
    finite_log_distribution = log_distribution.masked_fill(~torch.isfinite(log_distribution), 0.0)
    entropy = -(probabilities * finite_log_distribution).sum(dim=-1)
    current = torch.where(policy_mask, current, torch.zeros_like(current))
    entropy = torch.where(policy_mask, entropy, torch.zeros_like(entropy))
    return current, old_log_probs, entropy, policy_mask


def chunked_grpo_backward(
    model: Any,
    groups: Sequence[PromptRollouts],
    advantages: torch.Tensor,
    *,
    temperature: float,
    clip_epsilon: float,
    chunk_size: int,
    vocab_size: int | None = None,
) -> dict[str, float]:
    """Accumulate gradients exactly as one token-normalized batch, within memory bounds."""
    if len(groups) != advantages.shape[0] or chunk_size < 1:
        raise ValueError("Prompt groups, advantages, or rollout chunk size are invalid")
    group_size = len(groups[0].completions) if groups else 0
    if group_size < 2 or any(len(group.completions) != group_size for group in groups):
        raise ValueError("Every GRPO prompt must retain one complete, equally sized completion group")
    if tuple(advantages.shape) != (len(groups), group_size):
        raise ValueError("Group advantages do not align with the retained completion identities")
    trajectories = [
        (prompt_index, completion_index, group.prompt_ids, completion)
        for prompt_index, group in enumerate(groups)
        for completion_index, completion in enumerate(group.completions)
    ]
    total_actions = sum(completion.policy_tokens for *_prefix, completion in trajectories)
    if total_actions == 0:
        raise ValueError("GRPO update contains no policy actions")
    model.zero_grad(set_to_none=True)
    accum = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
    for start in range(0, len(trajectories), chunk_size):
        chunk = trajectories[start : start + chunk_size]
        current, old, entropy, mask = _batched_policy_statistics(
            model,
            [(prompt_ids, completion) for _prompt_index, _completion_index, prompt_ids, completion in chunk],
            temperature=temperature,
            vocab_size=vocab_size,
        )
        action_count = int(mask.sum().item())
        if action_count == 0:
            continue
        chunk_advantages = torch.stack(
            [advantages[prompt_index, completion_index] for prompt_index, completion_index, *_rest in chunk]
        ).to(dtype=torch.float32)[:, None]
        metrics = grpo_objective(
            current[:, None, :],
            old[:, None, :],
            chunk_advantages,
            mask[:, None, :],
            clip_epsilon=clip_epsilon,
            entropy=entropy[:, None, :],
        )
        (metrics.loss * (action_count / total_actions)).backward()
        for name in accum:
            accum[name] += float(getattr(metrics, name).detach().item()) * action_count
    return {name: value / total_actions for name, value in accum.items()} | {
        "policy_tokens": float(total_actions)
    }
