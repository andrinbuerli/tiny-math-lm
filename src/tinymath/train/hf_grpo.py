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
    entropy = -(log_distribution.exp() * log_distribution).sum(dim=-1)
    policy_mask = torch.tensor(rollout.policy_mask, dtype=torch.bool, device=device)
    old = torch.tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    if not (len(completion) == len(policy_mask) == len(old)):
        raise ValueError("Rollout token, action mask, and old-log-probability lengths disagree")
    current = torch.where(policy_mask, current, torch.zeros_like(current))
    entropy = torch.where(policy_mask, entropy, torch.zeros_like(entropy))
    return current, old, entropy, policy_mask


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
        for prompt_index, completion_index, prompt_ids, completion in trajectories[
            start : start + chunk_size
        ]:
            current, old, entropy, mask = native_policy_statistics(
                model, prompt_ids, completion, temperature=temperature, vocab_size=vocab_size
            )
            action_count = int(mask.sum().item())
            metrics = grpo_objective(
                current.view(1, 1, -1),
                old.view(1, 1, -1),
                advantages[prompt_index, completion_index].view(1, 1),
                mask.view(1, 1, -1),
                clip_epsilon=clip_epsilon,
                entropy=entropy.view(1, 1, -1),
            )
            (metrics.loss * (action_count / total_actions)).backward()
            for name in accum:
                accum[name] += float(getattr(metrics, name).detach().item()) * action_count
    return {name: value / total_actions for name, value in accum.items()} | {
        "policy_tokens": float(total_actions)
    }
