from __future__ import annotations

import sys
import types

import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.eval.qualitative import (
    _selected_rows,
    build_qualitative_table,
    log_qualitative_examples,
)


def test_qualitative_selection_stays_fixed_across_steps_and_row_order():
    rows = [
        {"question": f"{left} + {right}", "answer": str(left + right), "module": module}
        for module, offset in (("arithmetic__add_or_sub", 0), ("arithmetic__mixed", 10))
        for left, right in ((offset + 1, 1), (offset + 2, 1), (offset + 3, 1), (offset + 4, 1))
    ]

    at_5k = _selected_rows(rows, split="train", seed=23, per_module=2)
    at_10k = _selected_rows(list(reversed(rows)), split="train", seed=23, per_module=2)

    assert [(row["module"], row["question"]) for row in at_5k] == [
        (row["module"], row["question"]) for row in at_10k
    ]
    assert len(at_5k) == 4


def test_qualitative_table_covers_each_split_and_balances_modules(monkeypatch):
    rows = {
        "train": [
            {"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"},
            {"question": "1 + 2", "answer": "3", "module": "arithmetic__add_or_sub"},
            {"question": "2 + 1", "answer": "3", "module": "arithmetic__mixed"},
        ],
        "validation": [
            {"question": "2 + 2", "answer": "4", "module": "arithmetic__add_or_sub"},
        ],
        "test": [
            {"question": "3 + 1", "answer": "4", "module": "arithmetic__mixed"},
        ],
    }
    tokenizer = CharacterTokenizer.from_texts(
        format_prompt(row["question"]) + row["answer"]
        for examples in rows.values()
        for row in examples
    )

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(context_length=64)

        def generate(self, input_ids, **_kwargs):
            answer = torch.tensor(
                [[tokenizer.token_to_id["2"], tokenizer.eos_id]], device=input_ids.device
            )
            return torch.cat((input_ids, answer), dim=1)

    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Table=Table))

    class Run:
        def __init__(self):
            self.logs = []
            self.summary = {}

        def log(self, values, **kwargs):
            self.logs.append((values, kwargs))

    model = FixedModel()
    run = Run()
    count = log_qualitative_examples(
        model=model,
        tokenizer=tokenizer,
        split_rows=rows,
        run=run,
        training_arm="sft",
        seed=7,
        step=42,
        examples_per_module=1,
        max_new_tokens=8,
    )

    assert count == 4
    assert model.training
    table = run.logs[0][0]["qualitative/examples"]
    assert set(table.columns) == {
        "training_arm", "seed", "split", "module", "question", "target",
        "prediction", "completion", "predicted_scratchpad", "reference_scratchpad", "transcribed_scratchpad", "trace_structure_valid",
        "correct", "format_valid", "output_tokens", "capped",
    }
    assert {row[2] for row in table.data} == {"train", "validation", "test"}
    assert all(row[table.columns.index("reference_scratchpad")] == "" for row in table.data)
    assert {row[3] for row in table.data if row[2] == "train"} == {
        "arithmetic__add_or_sub", "arithmetic__mixed"
    }
    assert run.logs[0][0]["optimizer_step"] == 42
    assert run.logs[0][0]["train/greedy_accuracy"] in (0.0, 1.0)
    assert run.logs[0][0]["val/greedy_accuracy"] == 0.0
    assert run.logs[0][0]["test/greedy_accuracy"] == 0.0
    assert run.logs[0][1] == {}
    assert run.summary["qualitative/example_count"] == 4


def test_reasoning_qualitative_uses_unprefixed_prompt_and_full_context_cap(monkeypatch):
    rows = {
        split: [
            {
                "question": "1 + 1",
                "answer": "2",
                "module": "arithmetic__add_or_sub",
                "scratchpad": "REFERENCE MUST NOT ENTER PROMPT",
            }
        ]
        for split in ("train", "validation", "test")
    }
    trace = "t0=+(1,1)=2\n1,1,0:2,0"
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: 1 + 1\n", trace, "\nA: 2"], protocol="reasoning_v1"
    )
    observed = []

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(context_length=64)

        def generate(self, input_ids, *, max_new_tokens, **_kwargs):
            observed.append((tokenizer.decode(input_ids[0].tolist()), max_new_tokens))
            output = tokenizer.encode_completion("2", scratchpad=trace)
            return torch.cat((input_ids, torch.tensor([output], device=input_ids.device)), dim=1)

    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Table=Table))
    table, count, accuracy = build_qualitative_table(
        model=FixedModel(), tokenizer=tokenizer, split_rows=rows, training_arm="sft",
        seed=18, max_new_tokens=128, examples_per_module=1,
    )

    assert count == 3 and accuracy == {"train": 1.0, "validation": 1.0, "test": 1.0}
    assert all(prompt == "Q: 1 + 1\n" for prompt, _cap in observed)
    assert all(cap == 64 - len(tokenizer.encode("Q: 1 + 1\n", add_bos=True)) for _, cap in observed)
    assert table.data[0][8] == trace
    assert all(row[table.columns.index("reference_scratchpad")] == "REFERENCE MUST NOT ENTER PROMPT" for row in table.data)
    assert "Add 1 and 1" in table.data[0][table.columns.index("transcribed_scratchpad")]


def test_reasoning_qualitative_retains_predicted_scratchpad_when_answer_is_malformed(monkeypatch):
    rows = {
        split: [{"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"}]
        for split in ("train", "validation", "test")
    }
    trace = "t0=+(1,1)=2\n1,1,0:2,0"
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: 1 + 1\n", trace, "\nA: nope"], protocol="reasoning_v1"
    )

    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = types.SimpleNamespace(context_length=64)

        def generate(self, input_ids, **_kwargs):
            malformed = (
                [tokenizer.scp_id]
                + tokenizer.encode("\n" + trace + "\n")
                + [tokenizer.scp_end_id]
                + tokenizer.encode("\nA: nope")
                + [tokenizer.eos_id]
            )
            return torch.cat(
                (input_ids, torch.tensor([malformed], device=input_ids.device)), dim=1
            )

    class Table:
        def __init__(self, *, columns, data):
            self.columns = columns
            self.data = data

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Table=Table))
    table, count, _accuracy = build_qualitative_table(
        model=FixedModel(), tokenizer=tokenizer, split_rows=rows, training_arm="sft",
        seed=18, max_new_tokens=128, examples_per_module=1,
    )

    assert count == 3
    scratchpad_column = table.columns.index("predicted_scratchpad")
    validity_column = table.columns.index("trace_structure_valid")
    assert all(row[scratchpad_column] == trace for row in table.data)
    assert all(row[validity_column] is False for row in table.data)
