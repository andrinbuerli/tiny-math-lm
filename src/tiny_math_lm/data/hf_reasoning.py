"""Data identity and native-token formatting for the Qwen3 reasoning protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiny_math_lm.data.load import SELECTED_MODULES
from tiny_math_lm.data.reasoning import task_allows_calls
from tiny_math_lm.data.trace_explanations import transcribe_scratchpad
from tiny_math_lm.rewards.calls import CallError, validate_call
from tiny_math_lm.rewards.verify import parse_answer, verify_answer

PROMPT_FORMAT = "Q: {question}\n"
COMPLETION_FORMAT = "<think>\n{scratchpad}\n</think>\nA: {answer}"
FORMAT_VERSION = "exp16-qwen3-reasoning-v1"


@dataclass(frozen=True)
class NativeReasoningScore:
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


def parse_native_reasoning(text: str) -> tuple[str, str]:
    """Parse the complete native-tokenizer reasoning rendering, without token assumptions."""
    separator = "\n</think>\nA: "
    if (
        not text.startswith("<think>\n")
        or text.count("<think>") != 1
        or text.count("</think>") != 1
        or text.count(separator) != 1
    ):
        raise ValueError("Completion must contain one complete native thinking block")
    prefix, _separator, answer = text.partition(separator)
    scratchpad = prefix[len("<think>\n") :]
    if (
        not scratchpad.strip()
        or not answer
        or answer != answer.strip()
        or "\n" in answer
        or "\r" in answer
    ):
        raise ValueError("Completion has an empty scratchpad or invalid answer line")
    parse_answer(answer)
    return scratchpad, answer


def score_native_reasoning(
    text: str,
    row: dict[str, Any],
    *,
    capped: bool = False,
    failure: str | None = None,
    format_weight: float = 0.1,
    success_weight: float = 0.9,
    max_calls: int = 32,
    max_scalar_chars: int = 128,
) -> NativeReasoningScore:
    """Score decoded native-token text using the existing reasoning-v3 verifier."""
    if capped or failure:
        return NativeReasoningScore(call_policy_valid=failure != "disallowed_call")
    try:
        scratchpad, answer = parse_native_reasoning(text)
    except ValueError:
        return NativeReasoningScore()
    calls = [line for line in scratchpad.splitlines() if line.lstrip().startswith("call:")]
    correct_calls = 0
    policy_valid = not calls or task_allows_calls(row)
    for line in calls:
        try:
            validate_call(line, max_scalar_chars=max_scalar_chars)
            correct_calls += 1
        except CallError:
            pass
    try:
        trace_valid = bool(transcribe_scratchpad(scratchpad, version="reasoning_v3"))
    except ValueError:
        trace_valid = False
    trace_valid = trace_valid and correct_calls == len(calls) and len(calls) <= max_calls
    trace_valid = trace_valid and policy_valid
    answer_correct, answer_valid = verify_answer(
        answer, row["answer"], question=row["question"]
    )
    full_success = trace_valid and answer_correct and answer_valid
    format_reward = format_weight if policy_valid else 0.0
    success_reward = success_weight if full_success else 0.0
    return NativeReasoningScore(
        reward=round(format_reward + success_reward, 12),
        format_valid=True,
        trace_valid=trace_valid,
        answer_correct=answer_correct and answer_valid,
        full_success=full_success,
        scratchpad=scratchpad,
        answer=answer,
        reference_match=scratchpad == row.get("scratchpad"),
        call_count=len(calls),
        correct_calls=correct_calls,
        format_reward=format_reward,
        success_reward=success_reward,
        call_policy_valid=policy_valid,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_source_bundle(
    bundle_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_seed: int = 18,
) -> dict[str, Any]:
    """Fail closed unless the exact, previously verified reasoning-v3 bundle is used."""
    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reasoning-v3 manifest not found: {manifest_path}")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_sha256:
        raise ValueError(
            "Reasoning-v3 source manifest does not match Experiment 15 provenance "
            f"({manifest_hash} != {expected_manifest_sha256})"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("seed", -1)) != expected_seed:
        raise ValueError("Reasoning-v3 source bundle seed does not match Experiment 16")
    modules = set(manifest.get("selected_modules", []))
    if modules != set(SELECTED_MODULES):
        raise ValueError(f"Reasoning-v3 bundle has unexpected module inventory: {sorted(modules)}")
    if manifest.get("trace_protocol") != "reasoning_v3":
        raise ValueError("Experiment 16 requires the verified reasoning-v3 annotation protocol")
    return manifest


def select_training_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain every parent and only standalone subtasks marked eligible."""
    selected = []
    for row in rows:
        if row.get("is_sub_task", False):
            if row.get("training_eligible") is True:
                if row.get("overlapping_splits"):
                    raise ValueError("Eligible standalone subtask has a cross-pool overlap")
                selected.append(row)
        else:
            selected.append(row)
    if {row.get("module", "") for row in selected} - set(SELECTED_MODULES):
        raise ValueError("SFT rows contain a module outside the required five-module scope")
    return selected


def select_parent_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluation and prompt pools are parent-only; subtasks never enter evaluation."""
    selected = [row for row in rows if not row.get("is_sub_task", False)]
    if any(row.get("module") not in SELECTED_MODULES for row in selected):
        raise ValueError("Evaluation rows contain a module outside the required five-module scope")
    return selected


def identity_hash(rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    """Order-independent hash compatible with the Experiment 15 evaluation identity."""
    identities = [
        json.dumps(
            [row.get("module", ""), row.get("question", ""), row.get("answer", "")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    ]
    digest = hashlib.sha256()
    for identity in sorted(identities):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(identities)


def render_prompt(question: str) -> str:
    """Render the native-tokenizer prompt without rewriting original question text."""
    return PROMPT_FORMAT.format(question=question)


def render_completion(scratchpad: str, answer: str) -> str:
    """Render only the structural scratchpad boundaries with Qwen's native markers."""
    if not scratchpad.strip() or not answer or answer != answer.strip():
        raise ValueError("Reasoning completions require a scratchpad and scalar answer")
    if "\n" in answer or "\r" in answer:
        raise ValueError("Reasoning answer must occupy one line")
    return COMPLETION_FORMAT.format(scratchpad=scratchpad, answer=answer)


def encode_completion(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    context_length: int = 512,
    completion_cap: int = 512,
) -> tuple[list[int], list[int], int]:
    """Return input IDs, completion-only labels, and prompt length, including true EOS."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Qwen tokenizer must define a native EOS token")
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_id is None or end_think_id is None or think_id == tokenizer.unk_token_id:
        raise ValueError("Qwen native tokenizer is missing its existing thinking-token IDs")
    if tokenizer.encode("<think>", add_special_tokens=False) != [think_id]:
        raise ValueError("<think> is not one native tokenizer token")
    if tokenizer.encode("</think>", add_special_tokens=False) != [end_think_id]:
        raise ValueError("</think> is not one native tokenizer token")

    prompt = tokenizer.encode(render_prompt(str(row["question"])), add_special_tokens=False)
    completion_text = render_completion(str(row["scratchpad"]), str(row["answer"]))
    completion = tokenizer.encode(completion_text, add_special_tokens=False) + [int(eos_id)]
    if len(completion) > completion_cap:
        raise ValueError(
            f"Completion exceeds configured cap: {len(completion)} > {completion_cap}"
        )
    if len(prompt) + len(completion) > context_length:
        raise ValueError(
            "Complete eligible row exceeds context; refusing to truncate or drop it "
            f"(prompt={len(prompt)}, completion={len(completion)}, context={context_length})"
        )
    ids = [int(token) for token in (*prompt, *completion)]
    labels = [-100] * len(prompt) + [int(token) for token in completion]
    return ids, labels, len(prompt)
