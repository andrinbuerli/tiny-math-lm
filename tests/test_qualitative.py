from __future__ import annotations

import sys
import types

import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.eval.qualitative import log_qualitative_examples


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
        "prediction", "correct", "format_valid",
    }
    assert {row[2] for row in table.data} == {"train", "validation", "test"}
    assert {row[3] for row in table.data if row[2] == "train"} == {
        "arithmetic__add_or_sub", "arithmetic__mixed"
    }
    assert run.logs[0][0]["optimizer_step"] == 42
    assert run.logs[0][1] == {}
    assert run.summary["qualitative/example_count"] == 4
