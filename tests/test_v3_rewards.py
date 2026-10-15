from __future__ import annotations

import pytest

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.reasoning import (
    annotate_row,
    permitted_call_requests,
    task_allows_calls,
    verify_annotation,
)
from tinymath.rewards.calls import CallError, execute_call
from tinymath.rewards.reasoning import score_v3


def score(trace="step:+(1,2)\n=>3", answer="3", transform=lambda ids, t: ids, question="Calculate 1 + 2.", **kwargs):
    tokenizer = CharacterTokenizer.from_texts(
        ["<scp>\n</scp>\nA: " + trace + answer], protocol="reasoning_v3"
    )
    ids = transform(tokenizer.encode_completion(answer, scratchpad=trace), tokenizer)
    return score_v3(ids, tokenizer, {"question": question, "answer": "3"}, **kwargs)


def test_format_credit_and_optional_calls():
    assert score().reward == 1
    assert score().call_count == 0
    assert score(answer="4").reward == 0.1
    assert score(trace="step:unsupported").reward == 0.1
    assert score(trace="step:evaluate arithmetic operations in expression order\ncall:add(1,2) -> 3").reward == 0
    wrong_call = score(trace="step:evaluate arithmetic operations in expression order\ncall:add(1,2) -> 4")
    assert wrong_call.reward == 0 and wrong_call.answer_correct
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


@pytest.mark.parametrize(("question", "module", "answer"), [
    ("Work out 25.208953 - 6094639.", "arithmetic__add_or_sub", "-6094613.791047"),
    ("Calculate 25 + 32.", "arithmetic__add_or_sub", "57"),
    ("Calculate 25 * 32.", "arithmetic__mul", "800"),
    ("Calculate 25 / 32.", "arithmetic__div", "25/32"),
    ("Calculate (-53/50) - (-410201/100000).", "arithmetic__mixed", "304201/100000"),
])
def test_atomic_source_tasks_never_delegate(question, module, answer):
    row = {"question": question, "module": module, "answer": answer}
    assert not permitted_call_requests(row)
    assert not task_allows_calls(row)
    annotation = annotate_row(row, protocol="reasoning_v3")
    assert "call:" not in annotation["scratchpad"]
    assert annotation["tool_calls_allowed"] is False


def test_only_proper_subproblems_are_eligible_without_using_target_answer():
    row = {"question": "Calculate (1+2)*4.", "module": "arithmetic__mixed"}
    assert permitted_call_requests(row) == frozenset({"call:add(1,2) -> "})
    assert permitted_call_requests({**row, "answer": "deliberately unavailable"}) == permitted_call_requests(row)
    assert not permitted_call_requests({**row, "is_sub_task": True})
    assert not permitted_call_requests({"question": "unsupported task"})
    assert permitted_call_requests({"question": "Calculate (1+0)+0."}) == frozenset()
    assert task_allows_calls({"question": "Calculate (1+0)+0."})
    assert task_allows_calls({**row, "answer": "unavailable"})
    assert not task_allows_calls({**row, "is_sub_task": True})
    assert not task_allows_calls({"question": "unsupported task"})
    assert permitted_call_requests({"question": "Solve -21*o + 32 + 120 = -37 for o."}) == frozenset({"call:divide(-189,-21) -> "})


def test_reward_accepts_any_correct_call_for_compound_tasks():
    question = "Calculate (1+2)+0."
    prefix = "step:evaluate arithmetic operations in expression order\n"
    good = score(trace=prefix + "call:add(1,2) -> 3\nstep:+(3,0)", question=question)
    assert good.reward == 1 and good.call_policy_valid
    wrong = score(trace=prefix + "call:add(1,2) -> 4", question=question)
    assert wrong.reward == .1 and wrong.call_policy_valid
    for call in ("call:add(3,0) -> 3", "call:multiply(1,3) -> 3", "call:add(1,1) -> 2"):
        accepted = score(trace=prefix + call, question=question)
        assert accepted.reward == 1 and accepted.format_reward == .1
        assert accepted.format_valid and accepted.answer_correct
        assert accepted.call_policy_valid and accepted.trace_valid


def test_wrong_algebra_derivation_can_call_but_wrong_answer_gets_only_format_credit():
    row = {"question": "Solve 568*f + 662*f - 1137*f + 8928 = 0 for f.", "answer": "-96"}
    trace = "step:collect:1336f+(8928)=0f+(0)\nmove:1336f=-8928\ncall:divide(-8928,1336) -> -1116/167\nf=-1116/167"
    tokenizer = CharacterTokenizer.from_texts(["<scp>\n</scp>\nA: " + trace], protocol="reasoning_v3")
    result = score_v3(tokenizer.encode_completion("-1116/167", scratchpad=trace), tokenizer, row)
    assert result.call_policy_valid and result.correct_calls == 1
    assert not result.answer_correct and result.reward == .1


def test_screenshot_whole_subtraction_call_gets_zero_but_local_solution_gets_full_reward():
    row = {"question": "Work out 25.208953 - 6094639.", "answer": "-6094613.791047",
           "module": "arithmetic__add_or_sub"}
    bad_trace = "step:evaluate arithmetic operations in expression order\ncall:subtract(25208953/1000000,6094639) -> -6094613791047/1000000"
    local = annotate_row(row, protocol="reasoning_v3")
    tokenizer = CharacterTokenizer.from_texts(["<scp>\n</scp>\nA: " + bad_trace + local["scratchpad"] + row["answer"]], protocol="reasoning_v3")
    denied = score_v3(tokenizer.encode_completion(row["answer"], scratchpad=bad_trace), tokenizer, row)
    assert denied.reward == 0 and not denied.call_policy_valid
    assert denied.format_valid and denied.answer_correct
    assert score_v3(tokenizer.encode_completion(row["answer"], scratchpad=local["scratchpad"]), tokenizer, row).reward == 1


def test_fractional_scalar_recognition_preserves_left_associative_division():
    row = {"question": "Calculate 1/2/3/4.", "answer": "1/24", "module": "arithmetic__mixed"}
    annotation = annotate_row(row, protocol="reasoning_v3")
    assert permitted_call_requests(row) == frozenset({"call:divide(1,2) -> ", "call:divide(1/2,3) -> "})
    assert "call:divide(1/6,4)" not in annotation["scratchpad"]
    assert "step:/(1/6,4)" in annotation["scratchpad"]
    assert not permitted_call_requests({"question": "Calculate (1/2)/(3/4)."})


def test_v3_local_multiplication_uses_character_delimiter_budget():
    import string

    from tinymath.data.prompts import format_prompt

    row = {"question": "Product of -2 and 22252915.", "answer": "-44505830", "module": "arithmetic__mul"}
    tokenizer = CharacterTokenizer.from_texts([string.printable], protocol="reasoning_v3")

    def length(source, trace):
        return len(tokenizer.encode(format_prompt(source["question"], protocol="reasoning_v3"), add_bos=True)) + len(
            tokenizer.encode_completion(source["answer"], scratchpad=trace)
        )

    annotation = annotate_row(row, protocol="reasoning_v3", length_fn=length)
    assert length(row, annotation["scratchpad"]) <= 512
    assert annotate_row(row, protocol="reasoning_v3")["scratchpad"] == annotation["scratchpad"]
    verify_annotation(row, annotation)
    verify_annotation(row, annotation, length_fn=length)
