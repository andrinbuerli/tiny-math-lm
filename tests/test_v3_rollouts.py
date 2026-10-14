from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.generation import RolloutConfig, generate_rollouts
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.train.grpo import (
    completion_log_probs,
    group_outcome_rates,
    group_relative_advantages,
    grpo_objective,
)


class ScriptedModel(torch.nn.Module):
    def __init__(self, tokenizer, texts, cap=512):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(context_length=cap)
        self.scripts = [tokenizer.encode(text, add_eos=True) for text in texts]
        self.vocab = tokenizer.vocab_size
        self.eos = tokenizer.eos_id
        self.seen = []

    def forward_with_cache(self, ids, cache):
        position = 0 if cache is None else cache
        self.seen.append(ids.clone())
        logits = torch.full((len(ids), ids.shape[1], self.vocab), -100.0, device=ids.device)
        for lane, script in enumerate(self.scripts):
            logits[lane, -1, script[position] if position < len(script) else self.eos] = 100.0
        return logits, position + 1


def run_scripts(texts, *, config=None, cap=512, extra="", temperature=0):
    tokenizer = CharacterTokenizer.from_texts([*texts, extra], protocol="reasoning_v3")
    model = ScriptedModel(tokenizer, texts)
    result = generate_rollouts(
        model, torch.tensor([[tokenizer.bos_id]]), group_size=len(texts),
        max_completion_tokens=cap, temperature=temperature,
        eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id, tokenizer=tokenizer,
        rollout=config or RolloutConfig(executor="exact"),
    )
    return result, tokenizer, model


def test_independent_lanes_dependent_calls_and_observation_masks():
    texts = [
        "<scp>\nstep:evaluate arithmetic operations in expression order\ncall:add(1,2) -> 3\ncall:multiply(3,2) -> 6\n</scp>\nA: 6",
        "<scp>\nstep:+(1,2)\n=>3\n</scp>\nA: 3",
        "<scp>\nstep:evaluate arithmetic operations in expression order\ncall:divide(12,-1/2) -> -24\n</scp>\nA: -24",
    ]
    result, tokenizer, model = run_scripts(texts)
    assert result.termination_reasons == ("eos",) * 3
    assert [len(events) for events in result.call_events] == [2, 0, 1]
    for lane, text in enumerate(texts):
        ids = result.completion_ids[0, lane][result.completion_mask[0, lane]].tolist()
        assert tokenizer.decode(ids) == text
        assert ids[-1] == tokenizer.eos_id
    inserted = result.completion_mask & ~result.policy_mask
    assert inserted.sum() == 5
    assert (result.old_log_probs[inserted] == 0).all()
    assert (result.entropy[inserted] == 0).all()
    assert model.training
    assert all(ids.shape[1] == 1 for ids in model.seen[1:])
    # Every inserted character is fed back on the next cached forward pass.
    for lane, events in enumerate(result.call_events):
        for event in events:
            for position in range(event["result_start"], event["result_end"]):
                assert model.seen[position + 1][lane, 0] == result.completion_ids[0, lane, position]


@pytest.mark.parametrize(("line", "reason", "config"), [
    ("call:divide(1,0) -> ", "division_by_zero", RolloutConfig(executor="exact")),
    ("call:power(1,2) -> ", "malformed_request", RolloutConfig(executor="exact")),
    ("call:add(1,2)\n", "malformed_request", RolloutConfig(executor="exact")),
    ("call:multiply(99,99) -> ", "scalar_limit", RolloutConfig(executor="exact", max_scalar_chars=2)),
])
def test_executor_failures(line, reason, config):
    result, _, _ = run_scripts(["<scp>\nstep:x\n" + line], config=config)
    assert result.termination_reasons == (reason,)


def test_call_limit_and_unencodable_result():
    text = "<scp>\nstep:x\ncall:add(1,2) -> 3\ncall:add(1,2) -> "
    result, _, _ = run_scripts([text], config=RolloutConfig(executor="exact", max_calls=1))
    assert result.termination_reasons == ("call_limit",)
    result, _, _ = run_scripts(["<scp>\nstep:x\ncall:add(1,2) -> "])
    assert result.termination_reasons == ("unencodable_result",)


def test_detection_boundaries_and_executor_disabled():
    texts = ["call:add(1,2) -> 3", "<scp>\nstep:x\n</scp>\ncall:add(1,2) -> 3"]
    result, _, _ = run_scripts(texts)
    assert not any(result.call_events)
    result, _, _ = run_scripts(
        ["<scp>\nstep:x\ncall:add(1,2) -> 9\n</scp>\nA: 9"], config=RolloutConfig()
    )
    assert not any(result.call_events)
    assert torch.equal(result.policy_mask, result.completion_mask)


def test_budget_includes_insertions_and_eos():
    text = "<scp>\nstep:x\ncall:divide(12,-1/2) -> -24\n</scp>\nA: -24"
    prefix = text.index(" -> ") + 4
    result, _, _ = run_scripts([text], cap=prefix + 2)
    assert result.termination_reasons == ("budget_exceeded",)
    assert not result.call_events[0]
    result, _, _ = run_scripts([text], cap=len(text) + 1)
    assert result.termination_reasons == ("eos",)
    assert result.completion_mask.sum() == len(text) + 1
    result, _, _ = run_scripts([text], cap=len(text))
    assert result.termination_reasons == ("budget_exceeded",)


def test_cached_probabilities_seed_and_context_cap():
    torch.manual_seed(4)
    model = TinyMathGPT(GPTConfig(vocab_size=8, context_length=12, n_layers=1, n_heads=1, d_model=8, d_ff=16))
    model.eval()
    prompts = torch.tensor([[1, 4]])
    def sample():
        torch.manual_seed(9)
        return generate_rollouts(model, prompts, group_size=2, max_completion_tokens=30,
                                 temperature=0.8, eos_id=2, pad_id=0)
    result = sample()
    assert torch.equal(result.completion_ids, sample().completion_ids)
    current = completion_log_probs(model, prompts, result.completion_ids, temperature=0.8)
    assert torch.allclose(current[result.loss_mask], result.old_log_probs[result.loss_mask], atol=1e-6)
    assert result.completion_ids.shape[-1] <= 10


def test_fractional_groups_and_action_only_loss():
    rewards = torch.tensor([[0.1, 0.1], [0.1, 1.0], [0, 0], [1, 1]])
    advantages = group_relative_advantages(rewards)
    assert torch.equal(advantages[0], torch.zeros(2))
    rates = group_outcome_rates(rewards)
    assert all(rate.item() == 0.25 for rate in rates.values())
    current = torch.tensor([[[0.0, float("nan")], [0.0, float("inf")]]], requires_grad=True)
    old = torch.zeros_like(current)
    mask = torch.tensor([[[True, False], [True, False]]])
    metrics = grpo_objective(current, old, advantages[1:2], mask, reference_log_probs=old, kl_coefficient=0.1)
    metrics.loss.backward()
    assert torch.isfinite(metrics.loss)
    assert torch.isfinite(current.grad).all()
    assert (current.grad[~mask] == 0).all()


def test_real_model_cached_logs_and_conditioning_across_inserted_result(monkeypatch):
    text = "<scp>\nstep:x\ncall:add(1,2) -> 3\n</scp>\nA: 3"
    tokenizer = CharacterTokenizer.from_texts([text + "4"], protocol="reasoning_v3")
    model = TinyMathGPT(GPTConfig(vocab_size=tokenizer.vocab_size, context_length=128,
                                n_layers=1, n_heads=1, d_model=16, d_ff=32))
    model.eval()
    result_start = text.index(" -> ") + 4
    policy_ids = tokenizer.encode(text[:result_start] + text[result_start + 1:], add_eos=True)
    actions = iter(policy_ids)
    def scripted_sample(probabilities, num_samples):
        return torch.tensor([[next(actions)]], device=probabilities.device)
    monkeypatch.setattr(torch, "multinomial", scripted_sample)
    prompt = torch.tensor([[tokenizer.bos_id]])
    result = generate_rollouts(model, prompt, max_completion_tokens=100, temperature=0.8,
                               eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id, tokenizer=tokenizer,
                               rollout=RolloutConfig(executor="exact"))
    logs = completion_log_probs(model, prompt, result.completion_ids, temperature=0.8)
    assert torch.allclose(logs[result.loss_mask], result.old_log_probs[result.loss_mask], atol=1e-6)
    assert not result.loss_mask[0, 0, result_start]
    changed = result.completion_ids.clone()
    changed[0, 0, result_start] = tokenizer.token_to_id["4"]
    changed_logs = completion_log_probs(model, prompt, changed, temperature=0.8)
    assert not torch.allclose(logs[0, 0, result_start + 1], changed_logs[0, 0, result_start + 1], atol=1e-8)
