from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinymath.data.reasoning import annotate_row, verify_annotation
from tinymath.data.reasoning_v3_bundle import build_reasoning_v3_bundle
from tinymath.train.runner import _sft_training_rows


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
    train_rows = [json.loads(line) for line in (output / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()]
    test_rows = [json.loads(line) for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    parent = next(row for row in train_rows if not row["is_sub_task"])
    subtask = next(row for row in train_rows if row["is_sub_task"])
    heldout_subtask = next(row for row in test_rows if row["is_sub_task"])

    assert parent["trace_version"] == "reasoning_v3"
    assert parent["is_sub_task"] is False
    assert "call:add(25,32) -> 57" in parent["scratchpad"]
    assert subtask["question"] == "Calculate 25 + 32."
    assert subtask["answer"] == "57"
    assert subtask["trace_version"] == "reasoning_v2"
    assert "call:" not in subtask["scratchpad"]
    assert subtask["parent_row_index"] == 0 and subtask["call_index"] == 0
    assert subtask["call_operation"] == "add"
    assert subtask["call_operands"] == ["25", "32"]
    assert subtask["call_result"] == "57"
    assert subtask["is_sub_task"] is True
    assert subtask["training_eligible"] is False
    assert heldout_subtask["training_eligible"] is False
    assert result["splits"]["sft_train"]["examples"] == 2
    assert result["splits"]["sft_train"]["parent_examples"] == 1
    assert result["splits"]["sft_train"]["subtask_examples"] == 1
    assert result["annotation"]["subtask_schema_version"] == 2
    assert result["subtask_operand_semantics"] == "atomic_rational"
    assert not (output / "subtasks").exists()
    assert len(_sft_training_rows(train_rows)) == 1


def test_eligible_training_subtask_uses_character_tokenizer_and_enters_sft(workspace_tmpdir):
    source = workspace_tmpdir / "eligible-source"
    output = workspace_tmpdir / "eligible-v3"
    source.mkdir()
    _write_jsonl(source / "sft_train.jsonl", [{
        "question": "Calculate 25 + 32.", "answer": "57",
        "category": "train-easy", "module": "arithmetic__add_or_sub",
    }])
    (source / "manifest.json").write_text(json.dumps({
        "selected_modules": ["arithmetic__add_or_sub"],
        "splits": {"sft_train": {"path": "sft_train.jsonl", "examples": 1}},
    }), encoding="utf-8")

    build_reasoning_v3_bundle(source, output, workers=2)
    rows = [json.loads(line) for line in (output / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()]
    subtask = next(row for row in rows if row["is_sub_task"])

    assert subtask["training_eligible"] is True
    assert len(_sft_training_rows(rows)) == 2


def test_fraction_bundle_contains_one_operator_tasks_and_preserves_old_bundle(workspace_tmpdir):
    source = workspace_tmpdir / "source"
    source.mkdir()
    _write_jsonl(source / "sft_train.jsonl", [{
        "question": "Calculate -1.06 - -4.10201.", "answer": "3.04201",
        "module": "arithmetic__add_or_sub", "category": "train-easy",
    }])
    (source / "manifest.json").write_text(json.dumps({
        "splits": {"sft_train": {"path": "sft_train.jsonl", "examples": 1}},
    }))
    destination = workspace_tmpdir / "v3"
    destination.mkdir()
    (destination / "manifest.json").write_text("old bundle")
    progress = []
    build_reasoning_v3_bundle(source, destination, workers=1, progress=lambda *event: progress.append(event))
    assert {event[0] for event in progress} == {"scan", "vocabulary", "write"}
    backup, = workspace_tmpdir.glob(".v3.previous-*")
    assert (backup / "manifest.json").read_text() == "old bundle"
    rows = [json.loads(line) for line in (destination / "sft_train.jsonl").read_text().splitlines()]
    task = rows[1]
    assert task["question"] == "Calculate (-53/50) - (-410201/100000)."
    assert task["module"] == "arithmetic__add_or_sub"
    assert annotate_row(task, protocol="reasoning_v3")["scratchpad"].count("call:") == 1
    verify_annotation(task, task)


def test_failed_regeneration_never_modifies_published_bundle(workspace_tmpdir, monkeypatch):
    import tinymath.data.reasoning_v3_bundle as builder
    source = workspace_tmpdir / "source"
    source.mkdir()
    destination = workspace_tmpdir / "v3"
    destination.mkdir()
    (destination / "manifest.json").write_text("original")
    def fail(*args, **kwargs):
        raise ValueError("bad source row")
    monkeypatch.setattr(builder, "_build_reasoning_v3_bundle", fail)
    with pytest.raises(ValueError, match="bad source row"):
        build_reasoning_v3_bundle(source, destination)
    assert (destination / "manifest.json").read_text() == "original"
    assert not list(workspace_tmpdir.glob(".v3.building-*"))
