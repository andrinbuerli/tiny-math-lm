import json

import torch

from tiny_math_lm.data.hf_reasoning import NativeReasoningScore
from tiny_math_lm.eval import hf_qualitative
from tiny_math_lm.train.hf_generation import NativeRollout


def test_qualitative_selection_is_fixed_parent_only_and_module_balanced():
    rows = [
        {"module": "arithmetic__add_or_sub", "question": f"q{i}", "answer": "1"}
        for i in range(4)
    ]
    rows.append(
        {
            "module": "arithmetic__add_or_sub",
            "question": "subtask",
            "answer": "1",
            "is_sub_task": True,
        }
    )
    rows.extend(
        {"module": "arithmetic__mul", "question": f"m{i}", "answer": "2"}
        for i in range(3)
    )

    selected = hf_qualitative.select_validation_rows(
        rows, seed=18, examples_per_module=2
    )
    repeated = hf_qualitative.select_validation_rows(
        list(reversed(rows)), seed=18, examples_per_module=2
    )

    assert [(row["module"], row["question"]) for row in selected] == [
        (row["module"], row["question"]) for row in repeated
    ]
    assert len(selected) == 4
    assert all(row["question"] != "subtask" for row in selected)
    assert {row["module"] for row in selected} == {
        "arithmetic__add_or_sub",
        "arithmetic__mul",
    }


def test_native_qualitative_generation_scores_and_persists_validation_examples(
    tmp_path, monkeypatch
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class Tokenizer:
        eos_token_id = 99

        def encode(self, _text, add_special_tokens=False):
            return [1, 2]

        def decode(self, _tokens, **_kwargs):
            return "<think>\n2 + 2 = 4\n</think>\nA: 4"

    def fake_generate(*_args, **kwargs):
        assert kwargs["greedy"] is True
        assert kwargs["temperature"] == 0.0
        return NativeRollout(
            token_ids=[3],
            policy_mask=[True],
            old_log_probs=[0.0],
            entropy=[0.0],
            termination_reason="eos",
            call_events=(),
        )

    monkeypatch.setattr(hf_qualitative, "generate_native_rollout", fake_generate)
    monkeypatch.setattr(
        hf_qualitative,
        "score_native_reasoning",
        lambda *_args, **_kwargs: NativeReasoningScore(
            format_valid=True,
            trace_valid=True,
            answer_correct=True,
            full_success=True,
            scratchpad="2 + 2 = 4",
            answer="4",
            reward=1.0,
        ),
    )
    rows = [
        {
            "module": "arithmetic__add_or_sub",
            "question": "2 + 2",
            "answer": "4",
            "scratchpad": "2 + 2 = 4",
        }
    ]
    records = hf_qualitative.build_native_qualitative_examples(
        model=Model(),
        tokenizer=Tokenizer(),
        rows=rows,
        seed=18,
        optimizer_step=5000,
        context_length=512,
        max_new_tokens=128,
        examples_per_module=1,
        rollout_config={
            "executor": "exact",
            "format_weight": 0.1,
            "success_weight": 0.9,
        },
    )

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, values, step):
            self.logged.append((values, step))

    run = FakeRun()
    path = hf_qualitative.log_native_qualitative_examples(
        records=records, run=run, run_dir=tmp_path, wandb_step=5000
    )

    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert saved == records
    assert saved[0]["split"] == "validation"
    assert saved[0]["full_success"] is True
    assert "qualitative/examples" in run.logged[0][0]
    assert run.logged[0][1] == 5000


def test_greedy_validation_accuracy_uses_exact_answer_strings_and_a_mocked_generator(
    monkeypatch,
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class Tokenizer:
        eos_token_id = 99

        def encode(self, _text, add_special_tokens=False):
            return [1]

        def decode(self, token_ids, **_kwargs):
            return f"completion-{token_ids[0]}"

    def fake_generate(_model, _tokenizer, items, **kwargs):
        assert kwargs["greedy"] is True
        assert kwargs["temperature"] == 0.0
        assert len(items) == 2
        return tuple(
            NativeRollout(
                token_ids=[index + 1],
                policy_mask=[True],
                old_log_probs=[0.0],
                entropy=[0.0],
                termination_reason="eos",
                call_events=(),
            )
            for index in range(2)
        )

    monkeypatch.setattr(
        hf_qualitative, "generate_native_rollout_batch", fake_generate
    )
    monkeypatch.setattr(
        hf_qualitative,
        "score_native_reasoning",
        lambda text, _row, **_kwargs: NativeReasoningScore(
            format_valid=True, answer="4" if text.endswith("1") else "4.0"
        ),
    )
    rows = [
        {"module": "arithmetic__add_or_sub", "question": "2 + 2", "answer": "4"},
        {"module": "arithmetic__mul", "question": "2 * 2", "answer": "4"},
    ]

    accuracy, count = hf_qualitative.evaluate_native_greedy_exact_match(
        model=Model(),
        tokenizer=Tokenizer(),
        rows=rows,
        seed=18,
        context_length=512,
        max_new_tokens=128,
        batch_size=2,
        max_examples=2,
        rollout_config={"executor": "exact"},
    )

    assert accuracy == 0.5
    assert count == 2
