from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "filter_invalid_reasoning_bundle",
    Path(__file__).resolve().parents[1] / "scripts" / "filter_invalid_reasoning_bundle.py",
)
assert spec is not None and spec.loader is not None
filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_module)


def test_filter_drops_only_verified_matching_rows_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    good = {"module": "arithmetic__add_or_sub", "question": "1 + 1", "answer": "2"}
    bad = {
        "module": "arithmetic__mul",
        "question": "Calculate (1/2) * 4.",
        "answer": "2",
        "trace_version": "reasoning_v2",
        "is_sub_task": True,
        "scratchpad": "bad trace",
    }
    split_path = source / "sft_train.jsonl"
    split_path.write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n",
        encoding="utf-8",
    )
    original_split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    manifest = {
        "split_manifest_sha256": "source-splits",
        "splits": {
            "sft_train": {
                "path": split_path.name,
                "examples": 2,
                "parent_examples": 1,
                "subtask_examples": 1,
                "sha256": original_split_hash,
                "annotation_sha256": original_split_hash,
            }
        },
        "annotation": {"splits": {}},
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def verify(row, _annotation):
        if row.get("question") == bad["question"]:
            raise ValueError("expected verification failure")

    monkeypatch.setattr(filter_module, "verify_annotation", verify)
    destination = tmp_path / "derived"
    filtered = filter_module.filter_bundle(
        source=source,
        destination=destination,
        expected_manifest_sha256=original_manifest_hash,
        expected_error="expected verification failure",
        exclusions=[{
            "split": "sft_train",
            "line": 2,
            "module": bad["module"],
            "question": bad["question"],
            "answer": bad["answer"],
        }],
    )

    assert split_path.read_text(encoding="utf-8").count("\n") == 2
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == original_manifest_hash
    assert (destination / "sft_train.jsonl").read_text(encoding="utf-8") == json.dumps(good) + "\n"
    assert filtered["splits"]["sft_train"]["examples"] == 1
    assert filtered["splits"]["sft_train"]["subtask_examples"] == 0
    assert filtered["derived_filter"]["excluded_rows"][0]["line"] == 2
    assert filtered["source_split_manifest_sha256"] == "source-splits"
    assert filtered["split_manifest_sha256"] != "source-splits"
