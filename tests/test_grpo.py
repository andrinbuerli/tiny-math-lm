from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import torch

from tinymath.checkpoint import save_checkpoint
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.train import grpo_runner
from tinymath.train.grpo import (
    Rollouts,
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


def test_grpo_runner_scores_mixed_group_and_applies_gradient(workspace_tmpdir, monkeypatch):
    torch.manual_seed(11)
    bundle = workspace_tmpdir / "bundle"
    run_dir = workspace_tmpdir / "grpo"
    bundle.mkdir()
    tokenizer = CharacterTokenizer.from_texts([format_prompt("1 + 1") + "23"])
    tokenizer.save(bundle / "char_tokenizer.json")
    (bundle / "rlvr.jsonl").write_text(
        json.dumps({"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"}) + "\n",
        encoding="utf-8",
    )
    qualitative_row = json.dumps(
        {"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"}
    ) + "\n"
    (bundle / "sft_validation.jsonl").write_text(qualitative_row, encoding="utf-8")
    (bundle / "test.jsonl").write_text(qualitative_row, encoding="utf-8")
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size, context_length=64, n_layers=1,
        n_heads=2, d_model=16, d_ff=32,
    )
    initial_model = TinyMathGPT(model_config)
    checkpoint_path = workspace_tmpdir / "sft.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model": initial_model.state_dict(),
            "model_config": model_config.__dict__,
            "tokenizer_sha256": tokenizer.sha256,
            "step": 1,
        },
    )

    def fixed_rollouts(model, prompt_ids, *, temperature, **_kwargs):
        completion_ids = torch.tensor(
            [[[tokenizer.token_to_id["2"], tokenizer.eos_id],
              [tokenizer.token_to_id["3"], tokenizer.eos_id]]],
            device=prompt_ids.device,
        )
        mask = torch.ones_like(completion_ids, dtype=torch.bool)
        with torch.no_grad():
            old = grpo_runner.completion_log_probs(
                model, prompt_ids, completion_ids, temperature=temperature
            )
        return Rollouts(completion_ids, mask, old, torch.ones_like(old))

    monkeypatch.setattr(grpo_runner, "sample_group_rollouts", fixed_rollouts)
    config = SimpleNamespace(
        trainer=SimpleNamespace(bf16=False, weight_decay=0.0, grad_clip=1.0, save_every=1),
        rlvr=SimpleNamespace(
            updates=1, prompts_per_batch=1, group_size=2, temperature=1.0,
            max_completion_tokens=2, clip_epsilon=0.2, kl_coefficient=0.0, learning_rate=1e-2,
        ),
        tracking=SimpleNamespace(log_every=1),
        qualitative=SimpleNamespace(every_steps=1, examples_per_module=2),
    )

    class Run:
        def __init__(self):
            self.logged = []

        def log(self, values, **_kwargs):
            self.logged.append(values)

    run = Run()
    result = grpo_runner.train_grpo(
        config=config,
        bundle_dir=bundle,
        init_checkpoint=checkpoint_path,
        run_dir=run_dir,
        run=run,
    )
    saved = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    before = initial_model.state_dict()
    assert result["final_reward"] == 0.5
    assert result["qualitative_examples"] == 3
    inference = torch.load(result["inference_checkpoint"], map_location="cpu", weights_only=False)
    assert inference["update"] == 1
    assert inference["tokenizer_sha256"] == tokenizer.sha256
    assert inference["model"]
    assert "qualitative/examples" in run.logged[-1]
    assert any(not torch.equal(before[name], saved["model"][name]) for name in before)
    with (run_dir / "metrics.csv").open(encoding="utf-8") as handle:
        metrics = next(csv.DictReader(handle))
    assert float(metrics["grad_norm"]) > 0
    assert float(metrics["group/mixed_rate"]) == 1.0
    prompt_length = len(tokenizer.encode(format_prompt("1 + 1"), add_bos=True))
    expected_flops = initial_model.count_parameters() * (20 * prompt_length + 28)
    assert int(metrics["estimated_flops"]) == expected_flops
    config.rlvr.updates = 2
    resumed = grpo_runner.train_grpo(
        config=config,
        bundle_dir=bundle,
        init_checkpoint=checkpoint_path,
        run_dir=run_dir,
        run=Run(),
    )
    resumed_checkpoint = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert resumed["update"] == resumed_checkpoint["update"] == 2

