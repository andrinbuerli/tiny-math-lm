from __future__ import annotations

import torch

from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.train.grpo import (
    completion_log_probs,
    group_outcome_rates,
    group_relative_advantages,
    grpo_objective,
    sample_group_rollouts,
)


def test_group_advantages_zero_for_constant_groups_and_normalized_when_mixed():
    rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
    advantages = group_relative_advantages(rewards)
    assert torch.equal(advantages[0], torch.zeros(3))
    assert torch.allclose(advantages[1].mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.isfinite(advantages).all()


def test_reward_group_diagnostics_use_actual_binary_outcomes():
    rates = group_outcome_rates(torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]]))
    assert rates["mixed_group_rate"].item() == 0.5
    assert rates["all_correct_group_rate"].item() == 0.25
    assert rates["all_wrong_group_rate"].item() == 0.25


def test_clipped_policy_loss_masks_padding_and_has_finite_gradients():
    current = torch.tensor([[[0.1, 0.0], [-0.1, 0.15]]], requires_grad=True)
    old = torch.zeros_like(current)
    advantage = torch.tensor([[1.0, -1.0]])
    mask = torch.tensor([[[True, False], [True, True]]])
    entropy = torch.ones_like(current)
    reference = torch.full_like(current, -0.2)
    result = grpo_objective(
        current,
        old,
        advantage,
        mask,
        clip_epsilon=0.2,
        reference_log_probs=reference,
        kl_coefficient=0.1,
        entropy=entropy,
    )
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(current.grad).all()
    assert result.entropy.item() == 1.0
    assert result.clip_fraction.item() == 0.0
    assert current.grad[0, 0, 1].item() == 0.0


def test_rollouts_are_grouped_detached_and_keep_eos_in_the_mask():
    model = TinyMathGPT(
        GPTConfig(vocab_size=8, context_length=12, n_layers=1, n_heads=1, d_model=8, d_ff=16)
    )
    prompts = torch.tensor([[1, 4], [1, 5]])
    rollouts = sample_group_rollouts(
        model,
        prompts,
        group_size=4,
        max_completion_tokens=3,
        temperature=0.8,
        eos_id=2,
        pad_id=0,
    )
    assert rollouts.completion_ids.shape[:2] == (2, 4)
    assert rollouts.completion_mask.shape == rollouts.old_log_probs.shape
    assert not rollouts.old_log_probs.requires_grad
    assert model.training
    current = completion_log_probs(
        model, prompts, rollouts.completion_ids, temperature=0.8
    )
    assert current.shape == rollouts.old_log_probs.shape
    assert current.requires_grad

