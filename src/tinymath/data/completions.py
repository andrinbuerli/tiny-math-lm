"""Structural parsing for generated reasoning completion protocols."""

from __future__ import annotations

from dataclasses import dataclass

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.rewards.verify import AnswerFormatError, parse_answer


@dataclass(frozen=True)
class ReasoningCompletion:
    scratchpad: str
    answer: str


def parse_reasoning_completion(
    token_ids: list[int] | tuple[int, ...], tokenizer: CharacterTokenizer
) -> ReasoningCompletion:
    """Parse exactly one complete scratchpad, scalar answer line, and terminal EOS.

    The parser operates on reserved structural IDs before decoding ordinary text so
    question text that happens to spell a marker cannot affect completion structure.
    """
    if tokenizer.protocol not in {"reasoning_v1", "reasoning_v2", "reasoning_v3", "reasoning_v4"}:
        raise ValueError("Reasoning completions require a reasoning tokenizer")
    ids = list(token_ids)
    if not ids or ids[-1] != tokenizer.eos_id:
        raise ValueError("Reasoning completion must end with EOS")
    body = ids[:-1]
    if any(token in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.unk_id) for token in body):
        raise ValueError("Reasoning completion contains a reserved token outside its protocol")
    if tokenizer.protocol in {"reasoning_v3", "reasoning_v4"}:
        if tokenizer.eos_id in body:
            raise ValueError("Reasoning completion contains an interior EOS")
        rendered = tokenizer.decode(body)
        prefix, separator, suffix = rendered.partition("\n</scp>\nA: ")
        if (not separator or not prefix.startswith("<scp>\n")
                or rendered.count("<scp>") != 1 or rendered.count("</scp>") != 1
                or "A:" in prefix):
            raise ValueError(f"{tokenizer.protocol} completion needs one character-encoded scratchpad block")
        scratchpad = prefix[len("<scp>\n"):]
        answer = suffix
        if not scratchpad.strip() or not answer or answer != answer.strip() or "\n" in answer or "\r" in answer:
            raise ValueError("Reasoning completion has a missing, multiline, or trailing answer")
        try:
            parse_answer(answer)
        except AnswerFormatError as exc:
            raise ValueError("Reasoning completion answer is not a valid scalar") from exc
        return ReasoningCompletion(scratchpad=scratchpad, answer=answer)
    starts = [i for i, token in enumerate(body) if token == tokenizer.scp_id]
    ends = [i for i, token in enumerate(body) if token == tokenizer.scp_end_id]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("Reasoning completion must contain exactly one ordered scratchpad block")
    start, end = starts[0], ends[0]
    if start != 0 or any(
        token in (tokenizer.scp_id, tokenizer.scp_end_id) for token in body[start + 1 : end]
    ):
        raise ValueError("Reasoning completion contains duplicate or misplaced scratchpad markers")
    scratchpad = tokenizer.decode(body[start + 1 : end]).strip()
    suffix = tokenizer.decode(body[end + 1 :])
    if not scratchpad or not suffix.startswith("\nA: "):
        raise ValueError("Reasoning completion needs a nonempty scratchpad and terminal A: line")
    if tokenizer.protocol == "reasoning_v2" and not scratchpad.startswith("step:"):
        raise ValueError("reasoning_v2 scratchpads must begin with a derivation step")
    if tokenizer.protocol == "reasoning_v2" and "A:" in scratchpad:
        raise ValueError("reasoning_v2 scratchpads cannot contain an answer field")
    answer = suffix[len("\nA: ") :]
    if not answer or answer != answer.strip() or "\n" in answer or "\r" in answer:
        raise ValueError("Reasoning completion has a missing, multiline, or trailing answer")
    try:
        parse_answer(answer)
    except AnswerFormatError as exc:
        raise ValueError("Reasoning completion answer is not a valid scalar") from exc
    return ReasoningCompletion(scratchpad=scratchpad, answer=answer)
