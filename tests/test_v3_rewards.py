from __future__ import annotations

import pytest

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.rewards.calls import CallError, execute_call
from tinymath.rewards.reasoning import score_v3


def score(trace="step:+(1,2)\n=>3", answer="3", transform=lambda ids, t: ids, **kwargs):
    tokenizer = CharacterTokenizer.from_texts(
        ["<scp>\n</scp>\nA: " + trace + answer], protocol="reasoning_v3"
    )
    ids = transform(tokenizer.encode_completion(answer, scratchpad=trace), tokenizer)
    return score_v3(ids, tokenizer, {"question": "Calculate 1 + 2.", "answer": "3"}, **kwargs)


def test_format_credit_and_optional_calls():
    assert score().reward == 1
    assert score().call_count == 0
    assert score(answer="4").reward == 0.1
    assert score(trace="step:unsupported").reward == 0.1
    assert score(trace="step:evaluate arithmetic operations in expression order\ncall:add(1,2) -> 3").reward == 1
    wrong_call = score(trace="step:evaluate arithmetic operations in expression order\ncall:add(1,2) -> 4")
    assert wrong_call.reward == 0.1 and wrong_call.answer_correct
    assert wrong_call.correct_calls == 0 and not wrong_call.trace_valid


@pytest.mark.parametrize("transform", [
    lambda ids, t: ids[:-1],
    lambda ids, t: [t.eos_id] + ids,
    lambda ids, t: t.encode("<scp>\n") + ids,
    lambda ids, t: ids[:-1] + t.encode("\n</scp>") + [t.eos_id],
    lambda ids, t: ids[:-1] + t.encode("\nA: 3") + [t.eos_id],
])
def test_malformed_completions_get_no_format_credit(transform):
    assert score(transform=transform).reward == 0


def test_capped_failed_and_alternative_solutions():
    assert score(capped=True).reward == 0
    assert score(failure="division_by_zero").reward == 0
    assert score(trace="step:+(2,1)\n=>3", answer="6/2").reward == 1


@pytest.mark.parametrize(("call_text", "expected"), [
    ("call:add(-1,2) -> ", "1"),
    ("call:subtract(1/2,2/3) -> ", "-1/6"),
    ("call:multiply(-2/3,3) -> ", "-2"),
    ("call:divide(12,-1/2) -> ", "-24"),
])
def test_exact_operations(call_text, expected):
    assert execute_call(call_text).result == expected


@pytest.mark.parametrize("call_text", [
    "call:divide(1,0) -> ", "call:power(1,2) -> ",
    "call:add(2/4,1) -> ", "call:add(__import__('os'),1) -> ",
])
def test_invalid_calls(call_text):
    with pytest.raises(CallError):
        execute_call(call_text)


def test_scalar_limit():
    with pytest.raises(CallError, match="scalar_limit"):
        execute_call("call:multiply(99,99) -> ", max_scalar_chars=2)
