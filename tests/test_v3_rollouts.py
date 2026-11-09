from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tiny_math_lm.data.char_tokenizer import CharacterTokenizer
from tiny_math_lm.generation import RolloutConfig, generate_rollouts
from tiny_math_lm.model.gpt import GPTConfig, TinyMathGPT
from tiny_math_lm.train.grpo import (
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

    def forward_with_cache(self, ids, cache, attention_mask=None, position_ids=None):
        position = 0 if cache is None else cache
        self.seen.append(ids.clone())
        logits = torch.full((len(ids), ids.shape[1], self.vocab), -100.0, device=ids.device)
        for lane, script in enumerate(self.scripts):
            logits[lane, -1, script[position] if position < len(script) else self.eos] = 100.0
        return logits, position + 1


def run_scripts(texts, *, config=None, cap=512, extra="", temperature=0, question="Calculate ((1+2)*2)+(12/(-1/2))."):
    prompt_text = f"Q: {question}\n"
    tokenizer = CharacterTokenizer.from_texts([*texts, extra, prompt_text], protocol="reasoning_v3")
    model = ScriptedModel(tokenizer, texts)
    result = generate_rollouts(
        model, torch.tensor([tokenizer.encode(prompt_text, add_bos=True)]), group_size=len(texts),
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


def test_left_padded_rollout_keeps_prompt_alignment_and_exact_executor_events():
    questions = ["Calculate (1+2)+0.", "Calculate 1+2."]
    texts = [
        "<scp>\nstep:x\ncall:add(1,2) -> 3\n</scp>\nA: 3",
        "<scp>\nstep:x\n</scp>\nA: 3",
    ]
    prompt_texts = [f"Q: {question}\n" for question in questions]
    tokenizer = CharacterTokenizer.from_texts([*texts, *prompt_texts], protocol="reasoning_v3")
    encoded = [tokenizer.encode(text, add_bos=True) for text in prompt_texts]
    width = max(map(len, encoded))
    prompts = torch.full((len(encoded), width), tokenizer.pad_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompts, dtype=torch.bool)
    for index, ids in enumerate(encoded):
        prompts[index, -len(ids):] = torch.tensor(ids)
        prompt_mask[index, -len(ids):] = True
    rows = [{"question": question, "tool_calls_allowed": index == 0}
            for index, question in enumerate(questions)]
    model = ScriptedModel(tokenizer, texts)
    result = generate_rollouts(
        model, prompts, group_size=1, max_completion_tokens=64, temperature=0,
        eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id, tokenizer=tokenizer,
        rollout=RolloutConfig(executor="exact"), task_rows=rows, prompt_mask=prompt_mask,
    )
    assert result.termination_reasons == ("eos", "eos")
    assert [len(events) for events in result.call_events] == [1, 0]
    assert model.seen[0].shape == prompts.shape
    assert tokenizer.decode(result.completion_ids[0, 0].tolist()) == texts[0]
    assert tokenizer.decode(result.completion_ids[1, 0].tolist()) == texts[1]


def test_left_pad_uses_one_64_lane_batch_for_mixed_prompt_lengths():
    questions = ["Calculate 1+2." + (" " * length) for length in range(8)]
    prompt_texts = [f"Q: {question}\n" for question in questions]
    tokenizer = CharacterTokenizer.from_texts([*prompt_texts], protocol="reasoning_v3")
    encoded = [tokenizer.encode(text, add_bos=True) for text in prompt_texts]
    width = max(map(len, encoded))
    prompts = torch.full((len(encoded), width), tokenizer.pad_id, dtype=torch.long)
    mask = torch.zeros_like(prompts, dtype=torch.bool)
    for index, ids in enumerate(encoded):
        prompts[index, -len(ids):] = torch.tensor(ids)
        mask[index, -len(ids):] = True
    model = ScriptedModel(tokenizer, [""] * 64, cap=128)
    result = generate_rollouts(
        model, prompts, group_size=8, max_completion_tokens=64, temperature=0,
        eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id, tokenizer=tokenizer,
        rollout=RolloutConfig(), prompt_mask=mask,
    )
    assert result.completion_ids.shape[:2] == (8, 8)
    assert model.seen[0].shape == (64, width)
    assert len(model.seen) == 1


def test_masked_cached_attention_matches_unpadded_prompt_logits_and_decode():
    torch.manual_seed(31)
    model = TinyMathGPT(
        GPTConfig(vocab_size=19, context_length=8, n_layers=1, n_heads=1, d_model=8, d_ff=16)
    ).eval()
    prompts = torch.tensor([[0, 0, 1, 5, 7], [1, 4, 8, 3, 6]])
    mask = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])
    logical_positions = (mask.long().cumsum(dim=1) - 1).masked_fill(~mask, 0)
    with torch.no_grad():
        padded_logits, padded_cache = model.forward_with_cache(
            prompts, attention_mask=mask, position_ids=logical_positions
        )
        reference_logits, reference_cache = model.forward_with_cache(prompts[0:1, 2:])
        assert torch.allclose(padded_logits[0, -1], reference_logits[0, -1], atol=1e-6, rtol=1e-5)
        next_token = torch.tensor([[9]])
        padded_mask = torch.cat((mask[0:1], torch.ones((1, 1), dtype=torch.bool)), dim=1)
        padded_cache_single = tuple((key[0:1], value[0:1]) for key, value in padded_cache)
        decoded_logits, _ = model.forward_with_cache(
            next_token, padded_cache_single, attention_mask=padded_mask,
            position_ids=torch.tensor([[3]]),
        )
        reference_decoded_logits, _ = model.forward_with_cache(next_token, reference_cache)
        assert torch.allclose(
            decoded_logits[0, -1], reference_decoded_logits[0, -1], atol=1e-6, rtol=1e-5
        )
        cache = padded_cache
        growing_mask = mask.clone()
        for step in range(5):
            active = torch.tensor([True, step < 3])
            growing_mask = torch.cat((growing_mask, active[:, None]), dim=1)
            positions = torch.tensor([[3 + step], [min(5 + step, 7)]])
            _, cache = model.forward_with_cache(
                torch.tensor([[9], [9]]), cache, attention_mask=growing_mask,
                position_ids=positions,
            )
        assert cache[0][0].shape[-2] == 10  # physical left padding can exceed logical context


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


def test_completion_log_probs_ignores_only_masked_padding_past_context():
    torch.manual_seed(12)
    model = TinyMathGPT(
        GPTConfig(vocab_size=8, context_length=8, n_layers=1, n_heads=1, d_model=8, d_ff=16)
    ).eval()
    prompts = torch.tensor([[1, 4, 5, 6, 7, 2]])
    completions = torch.tensor([[[3, 4, 5, 6], [2, 3, 4, 5]]])
    mask = torch.tensor([[[True, True, False, False], [True, True, False, False]]])
    scored = completion_log_probs(
        model, prompts, completions, completion_mask=mask
    )
    assert scored.shape == completions.shape
    assert torch.isfinite(scored[..., :2]).all()
    assert torch.equal(scored[..., 2:], torch.zeros_like(scored[..., 2:]))
    with pytest.raises(ValueError, match="valid completion token exceeds"):
        completion_log_probs(
            model, prompts, completions, completion_mask=torch.ones_like(mask)
        )


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
    prompt_text = "Q: Calculate (1+2)*2.\n"
    tokenizer = CharacterTokenizer.from_texts([text + "4", prompt_text], protocol="reasoning_v3")
    model = TinyMathGPT(GPTConfig(vocab_size=tokenizer.vocab_size, context_length=128,
                                n_layers=1, n_heads=1, d_model=16, d_ff=32))
    model.eval()
    result_start = text.index(" -> ") + 4
    policy_ids = tokenizer.encode(text[:result_start] + text[result_start + 1:], add_eos=True)
    actions = iter(policy_ids)
    def scripted_sample(probabilities, num_samples):
        return torch.tensor([[next(actions)]], device=probabilities.device)
    monkeypatch.setattr(torch, "multinomial", scripted_sample)
    prompt = torch.tensor([tokenizer.encode(prompt_text, add_bos=True)])
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


@pytest.mark.parametrize("executor", ["exact", "none"])
def test_atomic_calls_are_rejected_before_execution(monkeypatch, executor):
    from tiny_math_lm import generation
    def forbidden(*args, **kwargs):
        raise AssertionError("A forbidden request reached the arithmetic executor")
    monkeypatch.setattr(generation, "execute_call", forbidden)
    for question, call in (
        ("Calculate 1+2.", "call:add(1,2) -> 3"),
        ("Calculate (-53/50)-(-410201/100000).", "call:subtract(-53/50,-410201/100000) -> 304201/100000"),
    ):
        result, _, _ = run_scripts(["<scp>\nstep:x\n" + call + "\n</scp>\nA: 3"],
            question=question, config=RolloutConfig(executor=executor))
        assert result.termination_reasons == ("disallowed_call",)
        assert result.call_events == ((),)
        assert (result.completion_mask & ~result.loss_mask).sum() == 0


@pytest.mark.parametrize("executor", ["exact", "none"])
@pytest.mark.parametrize("call", ["call:add(3,0) -> 3", "call:multiply(1,3) -> 3"])
def test_compound_tasks_accept_root_and_alternative_calls(executor, call):
    text = "<scp>\nstep:x\n" + call + "\n</scp>\nA: 3"
    result, tokenizer, _ = run_scripts([text], question="Calculate (1+2)+0.",
                                      config=RolloutConfig(executor=executor))
    assert result.termination_reasons == ("eos",)
    assert tokenizer.decode(result.completion_ids[0, 0].tolist()) == text
    assert len(result.call_events[0]) == int(executor == "exact")


def test_incorrect_algebra_coefficient_still_gets_exact_result():
    text = "<scp>\nstep:collect:1336f+(8928)=0f+(0)\nmove:1336f=-8928\ncall:divide(-8928,1336) -> -1116/167\n</scp>\nA: -1116/167"
    result, tokenizer, model = run_scripts([text],
        question="Solve 568*f + 662*f - 1137*f + 8928 = 0 for f.")
    assert result.termination_reasons == ("eos",)
    assert tokenizer.decode(result.completion_ids[0, 0].tolist()) == text
    event, = result.call_events[0]
    assert event["result"] == "-1116/167"
    assert (result.completion_mask & ~result.loss_mask).sum() == len("-1116/167")
    for position in range(event["result_start"], event["result_end"]):
        assert model.seen[position + 1][0, 0] == result.completion_ids[0, 0, position]


def test_disabled_executor_still_enforces_call_count():
    text = "<scp>\nstep:x\ncall:add(1,2) -> 3\ncall:add(99,2) -> 101"
    result, _, _ = run_scripts([text], config=RolloutConfig(executor="none", max_calls=1))
    assert result.termination_reasons == ("call_limit",)


def test_mixed_atomic_and_compound_prompt_lanes_have_independent_permissions():
    questions = ["Calculate 1 + 2.", "Calculate 1+2+0."]
    assert len(questions[0]) == len(questions[1])
    text = "<scp>\nstep:x\ncall:add(1,2) -> 3\n</scp>\nA: 3"
    prompts = [f"Q: {question}\n" for question in questions]
    tokenizer = CharacterTokenizer.from_texts([*prompts, text], protocol="reasoning_v3")
    model = ScriptedModel(tokenizer, [text, text])
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True) for prompt in prompts])
    result = generate_rollouts(model, ids, max_completion_tokens=100, temperature=0,
        eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id, tokenizer=tokenizer,
        rollout=RolloutConfig(executor="exact"), task_rows=[{"question": q} for q in questions])
    assert result.termination_reasons == ("disallowed_call", "eos")
    assert [len(events) for events in result.call_events] == [0, 1]
    with pytest.raises(ValueError, match="does not match"):
        generate_rollouts(model, ids, max_completion_tokens=100, temperature=0,
            eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id, tokenizer=tokenizer,
            task_rows=[{"question": questions[1]}, {"question": questions[0]}])


def test_eligible_call_still_enforces_scalar_resource_limit():
    result, _, _ = run_scripts(["<scp>\nstep:x\ncall:multiply(99,99) -> "],
        question="Calculate (99*99)+1.", config=RolloutConfig(executor="exact", max_scalar_chars=2))
    assert result.termination_reasons == ("scalar_limit",)
