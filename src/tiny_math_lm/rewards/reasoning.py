"""Shared v3 completion reward and independent correctness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tiny_math_lm.data.char_tokenizer import CharacterTokenizer
from tiny_math_lm.data.completions import parse_reasoning_completion
from tiny_math_lm.data.reasoning import task_allows_calls
from tiny_math_lm.data.trace_explanations import transcribe_scratchpad
from tiny_math_lm.rewards.calls import CallError, validate_call
from tiny_math_lm.rewards.verify import verify_answer

SCORER_VERSION = 3


@dataclass(frozen=True)
class V3Score:
    reward: float = 0.0
    format_valid: bool = False
    trace_valid: bool = False
    answer_correct: bool = False
    full_success: bool = False
    scratchpad: str = ""
    answer: str = ""
    reference_match: bool = False
    call_count: int = 0
    correct_calls: int = 0
    format_reward: float = 0.0
    success_reward: float = 0.0
    call_policy_valid: bool = True


def score_v3(
    token_ids: list[int], tokenizer: CharacterTokenizer, row: dict[str, Any], *,
    capped: bool = False, failure: str | None = None,
    format_weight: float = 0.1, success_weight: float = 0.9,
    max_calls: int = 32, max_scalar_chars: int = 128,
) -> V3Score:
    if tokenizer.protocol not in {"reasoning_v3", "reasoning_v4"}:
        raise ValueError("Call-based scoring requires a reasoning_v3 or reasoning_v4 tokenizer")
    if capped or failure:
        return V3Score(call_policy_valid=failure != "disallowed_call")
    try:
        parsed = parse_reasoning_completion(token_ids, tokenizer)
    except ValueError:
        return V3Score()
    lines = parsed.scratchpad.splitlines()
    calls = [line for line in lines if line.lstrip().startswith("call:")]
    correct_calls = 0
    policy_valid = not calls or task_allows_calls(row)
    for line in calls:
        try:
            validate_call(line, max_scalar_chars=max_scalar_chars)
            correct_calls += 1
        except CallError:
            pass
    try:
        trace_valid = bool(transcribe_scratchpad(parsed.scratchpad, version=tokenizer.protocol))
    except ValueError:
        trace_valid = False
    trace_valid = trace_valid and correct_calls == len(calls) and len(calls) <= max_calls
    correct, valid = verify_answer(parsed.answer, row["answer"], question=row["question"])
    trace_valid = trace_valid and policy_valid
    success = trace_valid and correct and valid
    success_reward = success_weight if success else 0.0
    return V3Score(
        reward=round(format_weight + success_reward, 12) if policy_valid else 0.0, format_valid=True,
        trace_valid=trace_valid, answer_correct=correct and valid, full_success=success,
        scratchpad=parsed.scratchpad, answer=parsed.answer,
        reference_match=parsed.scratchpad == row.get("scratchpad"),
        call_count=len(calls), correct_calls=correct_calls,
        format_reward=format_weight if policy_valid else 0.0, success_reward=success_reward,
        call_policy_valid=policy_valid,
    )
