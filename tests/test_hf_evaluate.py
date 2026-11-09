from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from omegaconf import OmegaConf

from tiny_math_lm.data.hf_reasoning import identity_hash
from tiny_math_lm.eval import hf_evaluate
from tiny_math_lm.eval.hf_evaluate import (
    _generate_qa,
    _generate_qa_batch,
    _summarize,
    _validate_checkpoint_identity,
)
from tiny_math_lm.train.hf_generation import NativeRollout


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


class _BatchCache:
    def batch_select_indices(self, indices):
        return None


def test_ordinary_qa_stops_at_first_line_even_when_token_crosses_boundary():
    answer, generated_tokens, capped = _generate_qa(
        _Model([1, 2, 3]), _Tokenizer(), "42 question", context=10, cap=8
    )
    assert answer == "42"
    assert generated_tokens == 3
    assert not capped


def test_ordinary_qa_stops_when_one_native_token_contains_the_line_break():
    class TokenizerWithEmbeddedLineBreak(_Tokenizer):
        id_to_text: ClassVar[dict[int, str]] = _Tokenizer.id_to_text | {
            5: "answer\ncontinued"
        }
        text_to_id: ClassVar[dict[str, int]] = {
            value: key for key, value in id_to_text.items()
        }

    answer, generated_tokens, capped = _generate_qa(
        _Model([5]), TokenizerWithEmbeddedLineBreak(), "question", context=10, cap=8
    )
    assert answer == "answer"
    assert generated_tokens == 1
    assert not capped


def test_ordinary_qa_does_not_search_prose_for_a_matching_scalar():
    answer, _, _ = _generate_qa(
        _Model([4, 0]), _Tokenizer(), "question", context=10, cap=8
    )
    assert answer == "wrong"


def test_batched_ordinary_qa_keeps_independent_eos_and_newline_stops():
    class BatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.steps = iter([[1, 4], [2, 0], [3, 0]])

        def forward(self, input_ids, **kwargs):
            token_ids = next(self.steps)[:input_ids.shape[0]]
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 10), -10.0)
            for index, token_id in enumerate(token_ids):
                logits[index, -1, token_id] = 10.0
            return SimpleNamespace(logits=logits, past_key_values=_BatchCache())

    results = _generate_qa_batch(
        BatchModel(), _Tokenizer(), ["first question", "second question"], context=10, cap=8
    )

    assert results == [("42", 3, False), ("wrong", 1, False)]


def test_batched_ordinary_qa_uses_left_padding_and_per_row_positions():
    class VaryingTokenizer(_Tokenizer):
        def encode(self, value, add_special_tokens=False):
            return [9] if "short" in value else [8, 9]

    class BatchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.steps = iter([[1, 4], [3, 0]])
            self.masks = []
            self.positions = []

        def forward(self, input_ids, attention_mask, position_ids, **kwargs):
            self.masks.append(attention_mask.tolist())
            self.positions.append(position_ids.tolist())
            token_ids = next(self.steps)
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 10), -10.0)
            for index, token_id in enumerate(token_ids):
                logits[index, -1, token_id] = 10.0
            return SimpleNamespace(logits=logits, past_key_values=_BatchCache())

    model = BatchModel()
    results = _generate_qa_batch(
        model, VaryingTokenizer(), ["short question", "longer question"], context=10, cap=8
    )

    assert results == [("4", 2, False), ("wrong", 1, False)]
    assert model.masks[0] == [[0, 1], [1, 1]]
    assert model.positions[0] == [[0, 0], [0, 1]]
    assert model.positions[1] == [[1], [2]]


def test_batched_ordinary_qa_preserves_results_when_some_prompts_exceed_context():
    class VaryingTokenizer(_Tokenizer):
        def encode(self, value, add_special_tokens=False):
            return [9] if "short" in value else [8, 9]

    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, **kwargs):
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], 10), -10.0)
            logits[:, -1, 0] = 10.0
            return SimpleNamespace(logits=logits, past_key_values=_BatchCache())

    results = _generate_qa_batch(
        EosModel(), VaryingTokenizer(), ["too long", "short"], context=2, cap=1
    )

    assert results == [("", 0, True), ("", 0, False)]


def test_reasoning_evaluation_batches_native_prompts_and_rows(monkeypatch):
    captured = {}
    rollout_result = NativeRollout(
        token_ids=[1, 0],
        policy_mask=[True, True],
        old_log_probs=[0.0, 0.0],
        entropy=[0.0, 0.0],
        termination_reason="eos",
        call_events=(),
    )
    expected = (
        rollout_result,
        rollout_result,
    )

    def generate_batch(model, tokenizer, items, **kwargs):
        captured["items"] = items
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(hf_evaluate, "generate_native_rollout_batch", generate_batch)
    rows = [
        {"question": "2 * 3", "module": "arithmetic__mul"},
        {"question": "x", "module": "arithmetic__mul"},
    ]
    result = hf_evaluate._generate_reasoning_batch(
        object(),
        _Tokenizer(),
        rows,
        context=512,
        cap=128,
        rollout={"executor": "exact"},
    )

    assert result == expected
    assert captured["items"] == [([2, 3], rows[0]), ([3], rows[1])]
    assert captured["kwargs"] == {
        "context_length": 512,
        "max_completion_tokens": 128,
        "temperature": 0.0,
        "rollout": {"executor": "exact"},
        "greedy": True,
    }


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


def test_checkpoint_validation_checks_revision_tokenizer_and_weight_hashes(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    weight_path = tmp_path / "model.safetensors"
    weight_path.write_bytes(b"fixture weights")
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"model.safetensors")
    digest.update(weight_hash.encode("ascii"))
    identity = {
        "model_revision": "revision",
        "model_config_sha256": "config",
        "tokenizer_sha256": "tokenizer",
        "parameters": 123,
        "checkpoint_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_weight_files": {"model.safetensors": weight_hash},
        "checkpoint_sha256": digest.hexdigest(),
    }
    (tmp_path / "experiment_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    provenance = {
        "model_revision": "revision",
        "model_config_sha256": "config",
        "tokenizer_sha256": "tokenizer",
        "model_parameters": 123,
    }
    assert _validate_checkpoint_identity(tmp_path, provenance)["checkpoint_sha256"] == digest.hexdigest()
    with pytest.raises(ValueError, match="model_revision"):
        _validate_checkpoint_identity(tmp_path, provenance | {"model_revision": "another"})


def test_evaluation_progress_discards_only_a_torn_final_append(tmp_path):
    progress = tmp_path / "progress.jsonl"
    identity = {"split": "test", "checkpoint": "base"}
    expected = {"row_index": 0, "question": "2 * 3"}
    hf_evaluate._progress_identity(progress, identity)
    hf_evaluate._append_progress(progress, expected)
    with progress.open("ab") as handle:
        handle.write(b'{"row_index": 1')

    recovered = hf_evaluate._progress_identity(progress, identity)
    assert recovered == [expected]
    assert progress.read_bytes().endswith(b"\n")
    hf_evaluate._append_progress(progress, {"row_index": 1, "question": "3 * 4"})
    assert len(hf_evaluate._progress_identity(progress, identity)) == 2


def test_evaluation_progress_rejects_corrupt_complete_rows(tmp_path):
    progress = tmp_path / "progress.jsonl"
    hf_evaluate._progress_identity(progress, {"split": "test"})
    progress.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt evaluation progress row 1"):
        hf_evaluate._progress_identity(progress, {"split": "test"})


def test_hf_evaluation_writes_both_formats_with_cpu_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    row = {
        "module": "arithmetic__mul",
        "question": "2 * 3",
        "answer": "6",
        "scratchpad": "step:multiply:2*3=6",
        "tool_calls_allowed": False,
        "is_sub_task": False,
    }
    identity, count = identity_hash([row])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("bundle manifest", encoding="utf-8")
    bundle_digest = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "source_bundle_manifest_sha256": bundle_digest,
                "splits": {
                    "test": {
                        "rows": count,
                        "fixed_question_identity_count": count,
                        "fixed_question_identity_sha256": identity,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class Split:
        def __init__(self, _cache_dir, _split):
            pass

        def __len__(self):
            return 1

        def rows(self):
            return iter([row])

    class Tokenizer(_Tokenizer):
        def encode(self, value, add_special_tokens=False):
            return [9]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, **kwargs):
            logits = torch.full((1, input_ids.shape[1], 10), -10.0)
            logits[:, -1, 0] = 10.0
            return SimpleNamespace(logits=logits, past_key_values=object())

    monkeypatch.setattr(hf_evaluate, "FIXED_SPLIT_COUNTS", {"test": 1})
    monkeypatch.setattr(hf_evaluate, "_cache_dir", lambda _config: cache)
    monkeypatch.setattr(hf_evaluate, "HFTokenSplit", Split)
    monkeypatch.setattr(hf_evaluate, "load_tokenizer", lambda _config: Tokenizer())
    monkeypatch.setattr(hf_evaluate, "load_model", lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(
        hf_evaluate,
        "model_provenance",
        lambda *_args: {
            "model_revision": "revision",
            "model_config_sha256": "config",
            "tokenizer_sha256": "tokenizer",
            "model_parameters": 1,
        },
    )
    monkeypatch.setattr(
        hf_evaluate,
        "generate_native_rollout_batch",
        lambda *_args, **_kwargs: (NativeRollout(
            token_ids=[1, 0],
            policy_mask=[True, True],
            old_log_probs=[0.0, 0.0],
            entropy=[0.0, 0.0],
            termination_reason="eos",
            call_events=(),
        ),),
    )
    config = OmegaConf.create(
        {
            "model": {"revision": "revision", "context_length": 512},
            "checkpoint_provenance": {
                "data_bundle_manifest_sha256": bundle_digest,
                "fixed_evaluation_question_identity_sha256": {"test": identity},
            },
            "eval": {"max_new_tokens": 8},
            "rollout": {"executor": "exact", "max_calls": 32, "max_scalar_chars": 128},
            "rlvr": {"reward_format_weight": 0.1, "reward_success_weight": 0.9},
            "experiment": {"name": "exp16_qwen3_base_reasoning_v3"},
        }
    )
    output = tmp_path / "eval"
    result = hf_evaluate.evaluate_hf_checkpoint(
        config=config,
        bundle_dir=bundle,
        output_dir=output,
        checkpoint=None,
        split_names=["test"],
    )
    assert set(result["splits"]) == {"test/ordinary_qa", "test/reasoning_executor"}
    assert (output / "test_ordinary_qa.csv").is_file()
    assert (output / "test_reasoning_executor.csv").is_file()
    assert Path(output / "results.json").is_file()
