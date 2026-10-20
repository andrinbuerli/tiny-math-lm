from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

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
        experiment="exp15_reasoning_v3_arithmetic_subtasks",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
        run_namespaces={
            "1m": {
                "sft": "fresh-sft-namespace",
                "continued_sft": "fresh-sft-namespace",
            }
        },
    )

    assert report["row_count"] == 27
    assert {row["status"] for row in report["rows"]} == {"not-run"}
    assert {row["source_arm"] for row in report["rows"] if row["arm"] == "sft"} == {
        "random_initialization"
    }
    assert {row["source_arm"] for row in report["rows"] if row["arm"] != "sft"} == {"sft"}
    assert {
        row["training_experiment"] for row in report["rows"]
        if row["model"] == "1m" and row["arm"] in {"sft", "continued_sft"}
    } == {"fresh-sft-namespace"}
    assert {item["status"] for item in report["compute_matching"].values()} == {"pending"}
    assert len(outputs) == 5
    assert all(path.exists() for path in outputs)
    markdown = (output / "intermediate_results.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Experiment 15 reasoning-v3 comparison")
    assert "| Model | Arm | Split | Baseline accuracy (scored) | +Reasoning accuracy (scored) | Δ percentage points | Status |" in markdown
    assert markdown.split("## Provenance", 1)[0].count("| 1m |") == 9
    assert "not-run" in markdown


def test_exp13_report_records_a_launch_blocker(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "annotation": {"protocol": "reasoning_v3"},
        "tokenizer": {},
        "splits": {},
    }), encoding="utf-8")
    blocker = "7 invalid multiplication subtask traces failed verification"
    output = workspace_tmpdir / "reports"
    report, _ = make_exp13_report(
        runs_dir=workspace_tmpdir / "runs",
        output_dir=output,
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
        launch_blocker=blocker,
    )

    assert report["launch_blocker"] == blocker
    assert all(row["status"] == "not-run" for row in report["rows"])
    assert all(row["delta_percentage_points"] is None for row in report["rows"])
    assert f"Fresh training and evaluation were not started: {blocker}" in report["limitations"][-1]
    markdown = (output / "intermediate_results.md").read_text(encoding="utf-8")
    assert "## Launch status" in markdown
    assert blocker in markdown


def test_exp13_report_records_a_per_arm_launch_gate(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "annotation": {"protocol": "reasoning_v3"},
        "tokenizer": {},
        "splits": {},
    }), encoding="utf-8")
    blocker = (
        "nvidia-smi reported 99% GPU utilization (21,118 MiB free of 24,564 MiB); "
        "the launch limit is 75%"
    )
    output = workspace_tmpdir / "reports"
    report, _ = make_exp13_report(
        runs_dir=workspace_tmpdir / "runs",
        output_dir=output,
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
        launch_blockers={"100m": {"grpo": blocker}},
    )

    grpo_rows = [row for row in report["rows"] if row["arm"] == "grpo"]
    assert {row["status"] for row in grpo_rows} == {"not-run"}
    assert {
        row["launch_blocker"] for row in grpo_rows if row["model"] == "100m"
    } == {blocker}
    assert {
        row["launch_blocker"] for row in grpo_rows if row["model"] in {"1m", "10m"}
    } == {None}
    assert report["launch_blockers"] == {"100m": {"grpo": blocker}}
    markdown = (output / "intermediate_results.md").read_text(encoding="utf-8")
    assert "## Per-arm launch gates" in markdown
    assert f"`100m/grpo` was not launched: {blocker}" in markdown


def test_exp13_report_includes_manifest_matched_source_trace_audit(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps({
        "annotation": {"protocol": "reasoning_v3"},
        "tokenizer": {},
        "splits": {},
    }), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    output = workspace_tmpdir / "reports"
    output.mkdir()
    audit = {
        "manifest_sha256": manifest_digest,
        "verifier": "tinymath.data.reasoning.verify_annotation",
        "checked_rows": 12,
        "split_counts": {"test": {"rows": 12, "parents": 8, "subtasks": 4, "sha256": "split-hash"}},
        "invalid_trace_rows": [{
            "split": "test", "line": 7, "module": "arithmetic__mul",
            "question": "Calculate 2 * 3.", "answer": "6",
        }],
    }
    (output / "bundle_validation.json").write_text(json.dumps(audit), encoding="utf-8")

    report, _ = make_exp13_report(
        runs_dir=workspace_tmpdir / "runs",
        output_dir=output,
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )

    assert report["bundle_validation"] == audit
    assert any("full source-bundle trace audit found 1 invalid rows" in item for item in report["limitations"])
    markdown = (output / "intermediate_results.md").read_text(encoding="utf-8")
    assert "## Source bundle trace audit" in markdown
    assert "`test` line 7" in markdown
    assert "Calculate 2 * 3." in markdown

    audit["manifest_sha256"] = "stale-manifest"
    stale_output = workspace_tmpdir / "reports-stale"
    stale_output.mkdir()
    (stale_output / "bundle_validation.json").write_text(json.dumps(audit), encoding="utf-8")
    stale_report, _ = make_exp13_report(
        runs_dir=workspace_tmpdir / "runs",
        output_dir=stale_output,
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )
    assert "bundle_validation" not in stale_report
    assert any("manifest hash does not match" in item for item in stale_report["limitations"])


def test_exp13_report_flags_training_bundle_manifest_mismatch(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    train_dir = runs / "exp13_reasoning_v3_arithmetic_subtasks" / "1m" / "sft" / "seed18"
    train_dir.mkdir(parents=True)
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "annotation": {"protocol": "reasoning_v3"}, "tokenizer": {}, "splits": {},
    }), encoding="utf-8")
    training_bundle = workspace_tmpdir / "filtered-bundle"
    training_bundle.mkdir()
    (training_bundle / "manifest.json").write_text(json.dumps({
        "derived_filter": {"excluded_rows": 7},
    }), encoding="utf-8")
    (train_dir / "resolved_config.json").write_text(json.dumps({
        "data_bundle": str(training_bundle),
    }), encoding="utf-8")

    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )

    row = next(row for row in report["rows"] if (
        row["model"], row["arm"], row["split"]
    ) == ("1m", "sft", "test"))
    assert row["training_data_bundle"] == str(training_bundle)
    assert row["training_bundle_manifest_sha256"] == hashlib.sha256(
        (training_bundle / "manifest.json").read_bytes()
    ).hexdigest()
    assert row["training_bundle_matches_reported_bundle"] is False
    assert any("bundle different" in item for item in report["limitations"])


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
    assert row["source_arm"] == "random_initialization"

    _write_predictions(eval_dir / "test_predictions.csv", list(reversed(rows)))
    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports-reordered",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
    )
    row = next(row for row in report["rows"] if (
        row["model"], row["arm"], row["split"]
    ) == ("1m", "sft", "test"))
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


def test_exp13_report_links_grpo_to_its_sft_checkpoint_and_run(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    experiment = "exp13_reasoning_v3_arithmetic_subtasks"
    grpo_experiment = f"{experiment}-grpo1m-fresh"
    sft_dir = runs / experiment / "1m" / "sft" / "seed18"
    sft_dir.mkdir(parents=True)
    source_sft = sft_dir / "best.pt"
    source_sft.write_bytes(b"selected-sft-checkpoint")
    (sft_dir / "wandb_run.json").write_text(json.dumps({
        "id": "sft-run-id",
        "mode": "online",
        "sync_status": "synced",
        "run_status": "finished",
        "url": "https://wandb.ai/entity/project/runs/sft-run-id",
        "entity": "entity",
        "project": "project",
        "identity": {"name": "sft-run", "group": "exp13-1m-sft"},
    }), encoding="utf-8")
    grpo_dir = runs / grpo_experiment / "1m" / "grpo" / "seed18"
    grpo_dir.mkdir(parents=True)
    grpo_checkpoint = grpo_dir / "latest.pt"
    grpo_checkpoint.write_bytes(b"grpo-checkpoint")
    eval_dir = runs / grpo_experiment / "1m" / "evaluation" / "grpo" / "latest" / "seed18"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results.json").write_text(json.dumps({
        "checkpoint": str(grpo_checkpoint.resolve()),
        "checkpoint_step": 100,
        "source_sft_checkpoint": str(source_sft.resolve()),
        "splits": {},
    }), encoding="utf-8")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "annotation": {}, "tokenizer": {}, "splits": {},
    }), encoding="utf-8")

    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
        experiment=experiment,
        run_namespaces={"1m": {"grpo": grpo_experiment}},
    )
    row = next(row for row in report["rows"] if (
        row["model"], row["arm"], row["split"]
    ) == ("1m", "grpo", "test"))
    assert row["source_arm"] == "sft"
    assert row["training_experiment"] == grpo_experiment
    assert row["checkpoint"] == str(grpo_checkpoint)
    assert row["source_sft_checkpoint"] == str(source_sft.resolve())
    assert row["source_sft_checkpoint_sha256"] == hashlib.sha256(
        source_sft.read_bytes()
    ).hexdigest()
    assert row["source_sft_wandb"]["run_id"] == "sft-run-id"


def test_exp13_report_preserves_source_sft_provenance_before_grpo_starts(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    source_sft = runs / "fresh-sft" / "100m" / "sft" / "seed18" / "best.pt"
    source_sft.parent.mkdir(parents=True)
    source_sft.write_bytes(b"selected-sft-checkpoint")
    (source_sft.parent / "wandb_run.json").write_text(json.dumps({
        "id": "100m-sft-run-id",
        "mode": "online",
        "sync_status": "synced",
        "run_status": "finished",
        "url": "https://wandb.ai/entity/project/runs/100m-sft-run-id",
        "entity": "entity",
        "project": "project",
        "identity": {"name": "sft-run", "group": "exp13-100m-sft"},
    }), encoding="utf-8")
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "annotation": {}, "tokenizer": {}, "splits": {},
    }), encoding="utf-8")

    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
        experiment="report-only",
        run_namespaces={
            "100m": {"sft": "fresh-sft", "grpo": "fresh-grpo"},
        },
    )

    rows = [row for row in report["rows"] if row["model"] == "100m" and row["arm"] == "grpo"]
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"not-run"}
    assert {row["checkpoint"] for row in rows} == {None}
    assert {row["checkpoint_sha256"] for row in rows} == {None}
    assert {row["source_sft_checkpoint"] for row in rows} == {str(source_sft.resolve())}
    assert {row["source_sft_checkpoint_sha256"] for row in rows} == {
        hashlib.sha256(source_sft.read_bytes()).hexdigest()
    }
    assert {row["source_sft_wandb"]["run_id"] for row in rows} == {"100m-sft-run-id"}


def test_exp13_compute_matching_uses_arm_namespaces_and_reports_checkpoint_update(workspace_tmpdir):
    baseline = _baseline_snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    sft_dir = runs / "fresh-sft" / "1m" / "sft" / "seed18"
    continued_dir = runs / "fresh-control" / "1m" / "continued_sft" / "seed18"
    grpo_dir = runs / "fresh-grpo" / "1m" / "grpo" / "seed18"
    sft_dir.mkdir(parents=True)
    continued_dir.mkdir(parents=True)
    grpo_dir.mkdir(parents=True)

    def write_metrics(path: Path, fieldnames: list[str], row: dict[str, int]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    write_metrics(
        sft_dir / "metrics.csv",
        ["optimizer_step", "estimated_flops", "additional_estimated_flops"],
        {"optimizer_step": 100_000, "estimated_flops": 1_000, "additional_estimated_flops": 0},
    )
    write_metrics(
        continued_dir / "metrics.csv",
        ["optimizer_step", "estimated_flops", "additional_estimated_flops"],
        {"optimizer_step": 80, "estimated_flops": 900, "additional_estimated_flops": 700},
    )
    write_metrics(
        grpo_dir / "metrics.csv",
        ["update", "estimated_flops"],
        {"update": 60, "estimated_flops": 2_000},
    )
    torch.save({"update": 50}, grpo_dir / "latest.pt")

    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"annotation": {}, "tokenizer": {}, "splits": {}}))
    report, _ = make_exp13_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        annotation_bundle_dir=bundle,
        experiment="report-only-namespace",
        run_namespaces={
            "1m": {
                "sft": "fresh-sft",
                "continued_sft": "fresh-control",
                "grpo": "fresh-grpo",
            }
        },
    )

    matching = report["compute_matching"]["1m"]
    assert matching["grpo_step"] == 60
    assert matching["grpo_estimated_flops"] == 2_000
    assert matching["continued_sft_step"] == 80
    assert matching["continued_sft_additional_estimated_flops"] == 700
    assert matching["continued_sft_to_grpo_flop_ratio"] == 0.35
    assert matching["status"] == "pending"
    row = next(row for row in report["rows"] if (
        row["model"], row["arm"], row["split"]
    ) == ("1m", "grpo", "test"))
    assert row["checkpoint_step"] == 50
