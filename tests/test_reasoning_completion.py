from __future__ import annotations

import pytest

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.completions import parse_reasoning_completion


@pytest.fixture
def tokenizer():
    return CharacterTokenizer.from_texts(
        ["\nA: 75", "+(47,28)\n7,8,0:5,1", "Q: marker text <scp>\n", "\nA: no"],
        protocol="reasoning_v1",
    )


def test_parse_reasoning_completion_requires_reserved_boundaries(tokenizer):
    ids = tokenizer.encode_completion("75", scratchpad="+(47,28)\n7,8,0:5,1")
    parsed = parse_reasoning_completion(ids, tokenizer)
    assert parsed.scratchpad == "+(47,28)\n7,8,0:5,1"
    assert parsed.answer == "75"


def test_malformed_or_unfinished_reasoning_outputs_are_rejected(tokenizer):
    complete = tokenizer.encode_completion("75", scratchpad="trace")
    malformed = [
        complete[:-1],  # capped output without EOS
        [tokenizer.scp_id] + complete,  # duplicate opening marker
        complete[:-1] + [tokenizer.scp_end_id, tokenizer.eos_id],  # duplicate close
        tokenizer.encode_completion("no", scratchpad="trace"),  # invalid scalar
        complete[:-1] + tokenizer.encode("\nA: 75") + [tokenizer.eos_id],  # duplicate answer line
    ]
    for ids in malformed:
        with pytest.raises(ValueError):
            parse_reasoning_completion(ids, tokenizer)


def test_question_marker_spelling_cannot_supply_completion_boundary(tokenizer):
    question = tokenizer.encode("Q: marker text <scp>\n")
    with pytest.raises(ValueError):
        parse_reasoning_completion(
            question + tokenizer.encode("75") + [tokenizer.eos_id], tokenizer
        )


def test_reasoning_v2_completion_keeps_answer_after_forward_trace():
    trace = "step:+(2,3)\n2,3,0:5,0"
    tokenizer = CharacterTokenizer.from_texts(
        [trace, "\nA: 5", "summary: 5", "step:+(2,3)\nA: 5"], protocol="reasoning_v2"
    )
    parsed = parse_reasoning_completion(
        tokenizer.encode_completion("5", scratchpad=trace), tokenizer
    )
    assert parsed.scratchpad == trace
    assert parsed.answer == "5"
    with pytest.raises(ValueError, match="begin with a derivation"):
        parse_reasoning_completion(
            tokenizer.encode_completion("5", scratchpad="summary: 5"), tokenizer
        )
    with pytest.raises(ValueError, match="cannot contain an answer field"):
        parse_reasoning_completion(
            tokenizer.encode_completion("5", scratchpad="step:+(2,3)\nA: 5"), tokenizer
        )
