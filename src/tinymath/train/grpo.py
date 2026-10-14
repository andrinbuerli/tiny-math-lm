"""GRPO rollout sampling, group advantages and clipped policy objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from tinymath.model.gpt import TinyMathGPT


@dataclass(frozen=True)
class Rollouts:
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class GRPOMetrics:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    entropy: torch.Tensor
    clip_fraction: torch.Tensor


@torch.no_grad()
def sample_group_rollouts(
    model: TinyMathGPT,
    prompt_ids: torch.Tensor,
    *,
    group_size: int,
    max_completion_tokens: int,
    temperature: float,
    eos_id: int,
    pad_id: int,
) -> Rollouts:
    """Sample independent completions without retaining an autograd graph."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] == 0 or prompt_ids.shape[1] == 0:
        raise ValueError("prompts must be a nonempty [batch, unpadded_sequence] tensor")
    if group_size < 2 or max_completion_tokens <= 0 or temperature <= 0:
        raise ValueError("group_size >= 2, positive completion length and temperature are required")
    was_training = model.training
    model.eval()
    batch, prompt_length = prompt_ids.shape
    sequences = prompt_ids[:, None, :].expand(batch, group_size, prompt_length).reshape(
        batch * group_size, prompt_length
    )
    active = torch.ones(batch * group_size, dtype=torch.bool, device=prompt_ids.device)
    completions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    cache = None
    try:
        for _ in range(max_completion_tokens):
            token_input = sequences if cache is None else sequences[:, -1:]
            logits, cache = model.forward_with_cache(token_input, cache)
            logits = logits[:, -1].float() / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            sampled = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            chosen_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            valid = active.clone()
            token = torch.where(valid, sampled, torch.full_like(sampled, pad_id))
            completions.append(token)
            masks.append(valid)
            old_log_probs.append(torch.where(valid, chosen_log_probs, torch.zeros_like(chosen_log_probs)))
            entropies.append(-(probabilities * log_probs).sum(dim=-1))
            active &= sampled.ne(eos_id)
            sequences = torch.cat((sequences, token.unsqueeze(-1)), dim=-1)
            if not torch.any(active):
                break
    finally:
        if was_training:
            model.train()
    completion_ids = torch.stack(completions, dim=-1).view(batch, group_size, -1)
    completion_mask = torch.stack(masks, dim=-1).view(batch, group_size, -1)
    old_log_probs_tensor = torch.stack(old_log_probs, dim=-1).view(batch, group_size, -1)
    entropy_tensor = torch.stack(entropies, dim=-1).view(batch, group_size, -1)
    return Rollouts(
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_log_probs=old_log_probs_tensor.detach(),
        entropy=entropy_tensor.detach(),
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
    """Report actual mixed, all-correct and all-wrong binary reward group rates."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [prompts, group_size >= 2]")
    correct = rewards.bool()
    all_correct = correct.all(dim=1)
    all_wrong = (~correct).all(dim=1)
    mixed = ~(all_correct | all_wrong)
    return {
        "mixed_group_rate": mixed.float().mean(),
        "all_correct_group_rate": all_correct.float().mean(),
        "all_wrong_group_rate": all_wrong.float().mean(),
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

    log_ratio = current_log_probs.float() - old_log_probs.detach().float()
    ratio = log_ratio.clamp(-20, 20).exp()
    expanded_advantage = advantages.detach().float().unsqueeze(-1)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * expanded_advantage
    token_policy_loss = -torch.minimum(unclipped, clipped)
    policy_loss = token_policy_loss[completion_mask].mean()
    if reference_log_probs is None:
        token_kl = torch.zeros_like(current_log_probs, dtype=torch.float32)
    else:
        # k3 estimator of KL(pi || reference), averaged over sampled current-policy tokens.
        ref_delta = reference_log_probs.detach().float() - current_log_probs.float()
        token_kl = ref_delta.exp() - ref_delta - 1
    kl = token_kl[completion_mask].mean()
    total_loss = policy_loss + kl_coefficient * kl
    token_entropy = entropy.float() if entropy is not None else torch.zeros_like(token_kl)
    if token_entropy.shape != current_log_probs.shape:
        raise ValueError("entropy, when provided, must match token log probability shape")
    mean_entropy = token_entropy[completion_mask].mean()
    clip_fraction = ((ratio - 1).abs() > clip_epsilon)[completion_mask].float().mean()
    return GRPOMetrics(
        loss=total_loss,
        policy_loss=policy_loss,
        kl=kl,
        entropy=mean_entropy,
        clip_fraction=clip_fraction,
    )
