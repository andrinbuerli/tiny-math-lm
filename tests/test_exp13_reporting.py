from __future__ import annotations

import csv
import json
from pathlib import Path

from tinymath.reporting import make_exp13_report


def _baseline_snapshot(path: Path) -> Path:
    measurements = [
        {
            "model": "1m", "arm": "sft", "split": "test",
            "accuracy": 0.5, "examples": 2,
        }
    ]
    provenance = {
        "baseline_evaluation_wandb_runs": {
            model: {
                arm: {"checkpoint_selector": "best" if arm != "grpo" else "latest"}
                for arm in ("sft", "grpo", "continued_sft")
            }
            for model in ("1m", "10m", "100m")
        }
    }
    path.write_text(json.dumps({
        "experiment": {"canonical_name": "frozen-baseline"},
        "frozen_at_utc": "2025-01-02T03:04:05+00:00",
        "available_measurements": measurements,
        "provenance": provenance,
    }), encoding="utf-8")
    return path


def _write_predictions(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["module", "question", "target", "prediction"])
        writer.writeheader()
        for module, question, target in rows:
            writer.writerow({"module": module, "question": question, "target": target, "prediction": target})


def test_exp13_report_lists_all_rows_and_marks_missing_evaluations_not_run(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "annotation": {"protocol": "reasoning_v3", "annotation_sha256": "annotation-hash"},
        "tokenizer": {"sha256": "tokenizer-hash"},
        "splits": {},
    }), encoding="utf-8")
    output = workspace_tmpdir / "reports"
    report, outputs = make_exp13_report(
        runs_dir=workspace_tmpdir / "runs",
        output_dir=output,
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )

    assert report["row_count"] == 27
    assert {row["status"] for row in report["rows"]} == {"not-run"}
    assert {item["status"] for item in report["compute_matching"].values()} == {"pending"}
    assert len(outputs) == 5
    assert all(path.exists() for path in outputs)
    markdown = (output / "intermediate_results.md").read_text(encoding="utf-8")
    assert "| Model | Arm | Split | Baseline accuracy (scored) | +Reasoning accuracy (scored) | Δ percentage points | Status |" in markdown
    assert markdown.split("## Provenance", 1)[0].count("| 1m |") == 9
    assert "not-run" in markdown


def test_exp13_delta_requires_complete_evaluation_and_identical_question_identity(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    baseline_predictions = runs / "frozen-baseline/1m/evaluation/sft/best/seed18/test_predictions.csv"
    rows = [
        ("arithmetic__add_or_sub", "1 + 1", "2"),
        ("arithmetic__mul", "2 × 3", "6"),
    ]
    _write_predictions(baseline_predictions, rows)
    checkpoint = runs / "exp13_reasoning_v3_arithmetic_subtasks/1m/sft/seed18/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    eval_dir = runs / "exp13_reasoning_v3_arithmetic_subtasks/1m/evaluation/sft/best/seed18"
    _write_predictions(eval_dir / "test_predictions.csv", rows)
    (eval_dir / "results.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_step": 100,
        "model_parameters": 1_000_000,
        "protocol": "reasoning_v3",
        "splits": {"test": {
            "examples": 2,
            "accuracy": 0.75,
            "format_validity": 1.0,
            "mean_output_tokens": 32,
            "module": {},
        }},
    }), encoding="utf-8")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"annotation": {}, "tokenizer": {}, "splits": {}}), encoding="utf-8")
    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )
    row = next(row for row in report["rows"] if (row["model"], row["arm"], row["split"]) == ("1m", "sft", "test"))
    assert row["status"] == "complete"
    assert row["same_question_identity"] is True
    assert row["delta_percentage_points"] == 25.0

    _write_predictions(eval_dir / "test_predictions.csv", rows[:1])
    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports-mismatch",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )
    row = next(row for row in report["rows"] if (row["model"], row["arm"], row["split"]) == ("1m", "sft", "test"))
    assert row["status"] == "complete"
    assert row["same_question_identity"] is False
    assert row["delta_percentage_points"] is None
