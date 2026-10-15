"""GRPO rollout sampling, group advantages and clipped policy objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.generation import RolloutConfig, Rollouts, generate_rollouts
from tinymath.model.gpt import TinyMathGPT


@dataclass(frozen=True)
class GRPOMetrics:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor
    clip_fraction: torch.Tensor


def sample_group_rollouts(
    model: TinyMathGPT, prompt_ids: torch.Tensor, *, group_size: int,
    max_completion_tokens: int, temperature: float, eos_id: int, pad_id: int,
    tokenizer: CharacterTokenizer | None = None, rollout: RolloutConfig | None = None,
    task_rows: list[dict] | None = None,
) -> Rollouts:
    """Sample independent groups with optional executor observations."""
    if group_size < 2 or temperature <= 0:
        raise ValueError("GRPO requires group_size >= 2 and positive temperature")
    return generate_rollouts(
        model, prompt_ids, group_size=group_size,
        max_completion_tokens=max_completion_tokens, temperature=temperature,
        eos_id=eos_id, pad_id=pad_id, tokenizer=tokenizer, rollout=rollout, task_rows=task_rows,
    )


def group_relative_advantages(rewards: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within each prompt group; constant groups receive zero."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [prompts, group_size >= 2]")
    values = rewards.float()
    means = values.mean(dim=1, keepdim=True)
    stds = values.std(dim=1, keepdim=True, unbiased=False)
    normalized = (values - means) / stds.clamp_min(eps)
    return torch.where(stds > eps, normalized, torch.zeros_like(normalized))


def group_outcome_rates(rewards: torch.Tensor) -> dict[str, torch.Tensor]:
    """Classify full-success, zero, format-only, and mixed-reward groups."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [prompts, group_size >= 2]")
    all_correct = (rewards == 1).all(dim=1)
    all_wrong = (rewards == 0).all(dim=1)
    all_format = ((rewards > 0) & (rewards < 1)).all(dim=1)
    mixed = (rewards != rewards[:, :1]).any(dim=1)
    return {
        "mixed_group_rate": mixed.float().mean(),
        "all_correct_group_rate": all_correct.float().mean(),
        "all_wrong_group_rate": all_wrong.float().mean(),
        "all_format_only_group_rate": all_format.float().mean(),
    }


def completion_log_probs(
    model: TinyMathGPT,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Recompute per-token policy log probabilities for `[B,G,T]` completions."""
    if prompt_ids.ndim != 2 or completion_ids.ndim != 3:
        raise ValueError("prompts must be [B,S] and completions [B,G,T]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, group_size, completion_length = completion_ids.shape
    if prompt_ids.shape[0] != batch or completion_length == 0:
        raise ValueError("prompt and completion batch dimensions must agree and completion must be nonempty")
    prompt_length = prompt_ids.shape[1]
    repeated_prompts = prompt_ids[:, None, :].expand(batch, group_size, prompt_length)
    sequences = torch.cat((repeated_prompts, completion_ids), dim=-1).reshape(
        batch * group_size, prompt_length + completion_length
    )
    logits = model(sequences)[:, prompt_length - 1 : prompt_length + completion_length - 1]
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    selected = log_probs.gather(-1, completion_ids.reshape(batch * group_size, completion_length, 1))
    return selected.squeeze(-1).view(batch, group_size, completion_length)


def grpo_objective(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_log_probs: torch.Tensor | None = None,
    kl_coefficient: float = 0.0,
    entropy: torch.Tensor | None = None,
) -> GRPOMetrics:
    """Compute token-level clipped GRPO loss and diagnostics with padding masks."""
    if current_log_probs.shape != old_log_probs.shape or current_log_probs.shape != completion_mask.shape:
        raise ValueError("current/old log probabilities and completion mask must have equal shapes")
    if current_log_probs.ndim != 3 or advantages.shape != current_log_probs.shape[:2]:
        raise ValueError("log probabilities must be [B,G,T] and advantages [B,G]")
    if completion_mask.dtype != torch.bool or not torch.any(completion_mask):
        raise ValueError("completion_mask must select generated tokens")
    if not 0 <= clip_epsilon < 1 or kl_coefficient < 0:
        raise ValueError("clip_epsilon must be in [0, 1) and kl_coefficient nonnegative")
    if reference_log_probs is not None and reference_log_probs.shape != current_log_probs.shape:
        raise ValueError("reference log probabilities must match current log probabilities")

    # Select actions before exponentiation: observations may contain arbitrary
    # placeholder logs and must contribute neither values nor gradients.
    current_actions = current_log_probs[completion_mask].float()
    old_actions = old_log_probs.detach()[completion_mask].float()
    log_ratio = current_actions - old_actions
    ratio = log_ratio.clamp(-20, 20).exp()
    expanded_advantage = advantages.detach().float().unsqueeze(-1).expand_as(current_log_probs)[completion_mask]
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * expanded_advantage
    token_policy_loss = -torch.minimum(unclipped, clipped)
    policy_loss = token_policy_loss.mean()
    if reference_log_probs is None:
        token_kl = torch.zeros_like(current_actions, dtype=torch.float32)
    else:
        # k3 estimator of KL(pi || reference), averaged over sampled current-policy tokens.
        ref_delta = reference_log_probs.detach()[completion_mask].float() - current_actions
        token_kl = ref_delta.exp() - ref_delta - 1
    kl = token_kl.mean()
    total_loss = policy_loss + kl_coefficient * kl
    if entropy is not None and entropy.shape != current_log_probs.shape:
        raise ValueError("entropy, when provided, must match token log probability shape")
    token_entropy = entropy[completion_mask].float() if entropy is not None else torch.zeros_like(token_kl)
    mean_entropy = token_entropy.mean()
    clip_fraction = ((ratio - 1).abs() > clip_epsilon).float().mean()
    return GRPOMetrics(
        loss=total_loss,
        policy_loss=policy_loss,
        kl=kl,
        entropy=mean_entropy,
        clip_fraction=clip_fraction,
    )
