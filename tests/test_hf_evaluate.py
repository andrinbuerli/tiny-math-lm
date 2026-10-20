from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import torch

from tinymath.eval.hf_evaluate import _generate_qa, _summarize


class _Tokenizer:
    eos_token_id = 0
    id_to_text: ClassVar[dict[int, str]] = {0: "<eos>", 1: "4", 2: "2", 3: "\n", 4: "wrong"}
    text_to_id: ClassVar[dict[str, int]] = {value: key for key, value in id_to_text.items()}

    def __len__(self):
        return 10

    def encode(self, value, add_special_tokens=False):
        if value in self.text_to_id:
            return [self.text_to_id[value]]
        return [self.text_to_id[character] for character in value if character in self.text_to_id] or [9]

    def decode(self, values, **kwargs):
        return "".join(self.id_to_text.get(value, "?") for value in values)


class _Model(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.values = iter(values)

    def forward(self, input_ids, **kwargs):
        logits = torch.full((1, input_ids.shape[1], 10), -10.0)
        logits[:, -1, next(self.values)] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_ordinary_qa_stops_at_first_line_even_when_token_crosses_boundary():
    answer, generated_tokens, capped = _generate_qa(
        _Model([1, 2, 3]), _Tokenizer(), "42 question", context=10, cap=8
    )
    assert answer == "42"
    assert generated_tokens == 3
    assert not capped


def test_ordinary_qa_does_not_search_prose_for_a_matching_scalar():
    answer, _, _ = _generate_qa(
        _Model([4, 0]), _Tokenizer(), "question", context=10, cap=8
    )
    assert answer == "wrong"


def test_hf_evaluation_summary_keeps_protocol_and_length_metrics():
    records = [
        {"module": "arithmetic__mul", "digit_width": 2, "operator_count": 1,
         "correct": True, "format_valid": True, "capped": False, "output_tokens": 3,
         "policy_tokens": 2, "executor_tokens": 1, "call_count": 1,
         "executor_failure": False, "full_success": True, "trace_valid": True},
        {"module": "arithmetic__mul", "digit_width": 3, "operator_count": 1,
         "correct": False, "format_valid": False, "capped": True, "output_tokens": 5,
         "policy_tokens": 5, "executor_tokens": 0, "call_count": 0,
         "executor_failure": True, "full_success": False, "trace_valid": False},
    ]
    result = _summarize(records)
    assert result["accuracy"] == 0.5
    assert result["by_module"]["arithmetic__mul"]["examples"] == 2
    assert result["capped_completions"] == 1
    assert result["policy_tokens"] == 7
    assert result["executor_tokens"] == 1
