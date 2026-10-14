from __future__ import annotations

import json
from pathlib import Path

from tinymath.data.reasoning_v3_bundle import build_reasoning_v3_bundle


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_reasoning_v3_bundle_writes_parent_and_linked_subtask_splits(workspace_tmpdir):
    source = workspace_tmpdir / "source"
    output = workspace_tmpdir / "v3"
    source.mkdir()
    example = {
        "question": "Calculate 25 + 32.", "answer": "57",
        "category": "train-easy", "module": "arithmetic__add_or_sub",
    }
    _write_jsonl(source / "sft_train.jsonl", [example])
    _write_jsonl(source / "test.jsonl", [example])
    (source / "manifest.json").write_text(json.dumps({
        "selected_modules": ["arithmetic__add_or_sub"],
        "splits": {
            "sft_train": {"path": "sft_train.jsonl", "examples": 1},
            "test": {"path": "test.jsonl", "examples": 1},
        },
    }), encoding="utf-8")

    result = build_reasoning_v3_bundle(source, output)
    parent = json.loads((output / "sft_train.jsonl").read_text(encoding="utf-8"))
    subtask = json.loads((output / "subtasks/sft_train.jsonl").read_text(encoding="utf-8"))
    heldout_subtask = json.loads((output / "subtasks/test.jsonl").read_text(encoding="utf-8"))

    assert parent["trace_version"] == "reasoning_v3"
    assert "call:add(25,32) -> 57" in parent["scratchpad"]
    assert subtask["question"] == "Calculate 25 + 32."
    assert subtask["answer"] == "57"
    assert subtask["trace_version"] == "reasoning_v2"
    assert "call:" not in subtask["scratchpad"]
    assert subtask["parent_row_index"] == 0 and subtask["call_index"] == 0
    assert subtask["call_operation"] == "add"
    assert subtask["call_operands"] == ["25", "32"]
    assert subtask["call_result"] == "57"
    assert subtask["training_eligible"] is False
    assert heldout_subtask["training_eligible"] is False
    assert result["splits"]["sft_train"]["examples"] == 1
    assert result["subtask_splits"]["sft_train"]["examples"] == 1
