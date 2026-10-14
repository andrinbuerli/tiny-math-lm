from __future__ import annotations

import csv
import json

from tinymath.reporting import make_exp11_report


def _snapshot(path, *, measurements=None, missing=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": {"canonical_name": "baseline"},
                "available_measurements": measurements or [],
                "coverage": {"missing": missing or {}},
                "provenance": {
                    "baseline_evaluation_wandb_runs": {
                        "1m": {"sft": {"checkpoint_selector": "best"}}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _predictions(path, questions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["module", "question", "target", "scratchpad", "output_tokens"],
        )
        writer.writeheader()
        for question, target in questions:
            writer.writerow(
                {
                    "module": "arithmetic__add_or_sub",
                    "question": question,
                    "target": target,
                    "scratchpad": "t0=1+1=2",
                    "output_tokens": 20,
                }
            )


def test_exp11_report_has_27_rows_and_not_run_states_without_results(workspace_tmpdir):
    baseline = _snapshot(workspace_tmpdir / "baseline.json")
    report_dir = workspace_tmpdir / "reports"
    report, outputs = make_exp11_report(
        runs_dir=workspace_tmpdir / "runs",
        output_dir=report_dir,
        baseline_snapshot=baseline,
        baseline_runs_dir=workspace_tmpdir / "runs",
        annotation_bundle_dir=workspace_tmpdir / "annotation",
    )

    assert report["row_count"] == 27
    assert {row["exp11_status"] for row in report["rows"]} == {"not-run"}
    assert {row["baseline_status"] for row in report["rows"]} == {"missing"}
    markdown = (report_dir / "intermediate_results.md").read_text(encoding="utf-8")
    main_table = markdown.split("## Approximate compute matching", 1)[0]
    assert main_table.count("| 1m |") == 9
    assert "not-run" in markdown
    assert "scratchpad tokens" in markdown
    assert "effective batch 128" in markdown
    assert len([path for path in outputs if path.suffix == ".png"]) == 2
    assert len([path for path in outputs if path.suffix == ".svg"]) == 2


def test_exp11_delta_requires_complete_same_question_identity(workspace_tmpdir):
    question_rows = [("1+1", "2"), ("2+3", "5")]
    baseline = _snapshot(
        workspace_tmpdir / "baseline.json",
        measurements=[
            {
                "model": "1m", "arm": "sft", "split": "test", "accuracy": 0.5,
                "examples": 2, "checkpoint_step": 2,
            }
        ],
    )
    runs = workspace_tmpdir / "runs"
    main_checkpoint = runs / "exp11_reasoning_500k_100k" / "1m" / "sft" / "seed18" / "best.pt"
    main_checkpoint.parent.mkdir(parents=True)
    main_checkpoint.write_bytes(b"main")
    baseline_csv = (
        runs / "baseline" / "1m" / "evaluation" / "sft" / "best" / "seed18" / "test_predictions.csv"
    )
    _predictions(baseline_csv, question_rows)
    evaluation = (
        runs / "exp11_reasoning_500k_100k" / "1m" / "evaluation" / "sft" / "best"
        / "step94000-0123456789ab" / "seed18"
    )
    evaluation.mkdir(parents=True)
    (evaluation / "results.json").write_text(
        json.dumps(
            {
                "checkpoint": str(main_checkpoint.resolve()),
                "checkpoint_step": 4,
                "splits": {
                    "test": {
                        "examples": 2, "accuracy": 0.75, "format_validity": 1.0,
                        "mean_output_tokens": 22, "capped_completions": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _predictions(evaluation / "test_predictions.csv", question_rows)

    report, _outputs = make_exp11_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        baseline_runs_dir=runs,
        annotation_bundle_dir=workspace_tmpdir / "annotation",
    )
    row = next(
        row for row in report["rows"]
        if (row["model"], row["arm"], row["split"]) == ("1m", "sft", "test")
    )
    assert row["baseline_status"] == "complete"
    assert row["exp11_status"] == "complete"
    assert row["same_question_identity"] is True
    assert row["delta_accuracy"] == 0.25

    _predictions(evaluation / "test_predictions.csv", [("9+9", "18"), ("8+7", "15")])
    report, _outputs = make_exp11_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports_mismatch",
        baseline_snapshot=baseline,
        baseline_runs_dir=runs,
        annotation_bundle_dir=workspace_tmpdir / "annotation",
    )
    row = next(
        row for row in report["rows"]
        if (row["model"], row["arm"], row["split"]) == ("1m", "sft", "test")
    )
    assert row["exp11_status"] == "complete"
    assert row["same_question_identity"] is False
    assert row["delta_accuracy"] is None


def test_exp11_running_and_partial_states_are_visible(workspace_tmpdir):
    baseline = _snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    running = runs / "exp11_reasoning_500k_100k" / "1m" / "evaluation" / "sft" / "best" / "seed18"
    running.mkdir(parents=True)
    main_sft_checkpoint = runs / "exp11_reasoning_500k_100k" / "1m" / "sft" / "seed18" / "best.pt"
    (running / "resolved_config.json").write_text(
        json.dumps({"checkpoint": str(main_sft_checkpoint.resolve())}), encoding="utf-8"
    )
    (running / "wandb_run.json").write_text(json.dumps({"run_status": "running"}), encoding="utf-8")
    partial = runs / "exp11_reasoning_500k_100k" / "10m" / "evaluation" / "grpo" / "latest" / "seed18"
    partial.mkdir(parents=True)
    main_grpo_checkpoint = runs / "exp11_reasoning_500k_100k" / "10m" / "grpo" / "seed18" / "latest.pt"
    (partial / "resolved_config.json").write_text(
        json.dumps({"checkpoint": str(main_grpo_checkpoint.resolve())}), encoding="utf-8"
    )
    (partial / "progress.json").write_text(
        json.dumps({"scored": {"test": [{"row_index": 0}]}}), encoding="utf-8"
    )

    report, _outputs = make_exp11_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        baseline_runs_dir=runs,
        annotation_bundle_dir=workspace_tmpdir / "annotation",
    )
    states = {(row["model"], row["arm"], row["split"]): row["exp11_status"] for row in report["rows"]}
    assert states[("1m", "sft", "test")] == "running"
    assert states[("10m", "grpo", "test")] == "partial"


def test_main_report_does_not_promote_pilot_evaluation(workspace_tmpdir):
    baseline = _snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    main_checkpoint = (
        runs / "exp11_reasoning_500k_100k" / "1m" / "sft" / "seed18" / "best.pt"
    )
    main_checkpoint.parent.mkdir(parents=True)
    main_checkpoint.write_bytes(b"main")
    (main_checkpoint.parent / "wandb_run.json").write_text(
        json.dumps({
            "id": "main-run-id", "mode": "offline", "sync_status": "unsynced",
            "url": "https://wandb.ai/unit/tinymath-rl/runs/main-run-id",
            "run_status": "running", "identity": {
                "project": "tinymath-rl", "group": "exp11-main",
                "name": "exp11-main-1m-sft-seed18",
                "metadata": {"phase": "main"},
            },
        }),
        encoding="utf-8",
    )
    (main_checkpoint.parent / "metrics.csv").write_text(
        "optimizer_step,tokens_seen,estimated_flops\n10,100,1000\n", encoding="utf-8"
    )
    baseline_metrics = runs / "baseline" / "1m" / "sft" / "seed18" / "metrics.csv"
    baseline_metrics.parent.mkdir(parents=True)
    baseline_metrics.write_text(
        "optimizer_step,tokens_seen,estimated_flops\n100,1000,10000\n", encoding="utf-8"
    )
    pilot_checkpoint = main_checkpoint.parent / "pilot-2000-examples-10-updates" / "seed18" / "best.pt"
    pilot_checkpoint.parent.mkdir(parents=True)
    pilot_checkpoint.write_bytes(b"pilot")
    pilot_eval = (
        runs / "exp11_reasoning_500k_100k" / "1m" / "evaluation" / "sft" / "best"
        / "step10-pilot" / "seed18"
    )
    pilot_eval.mkdir(parents=True)
    (pilot_eval / "resolved_config.json").write_text(
        json.dumps({"checkpoint": str(pilot_checkpoint.resolve())}), encoding="utf-8"
    )
    (pilot_eval / "results.json").write_text(
        json.dumps({"checkpoint": str(pilot_checkpoint.resolve()), "splits": {"test": {"examples": 20}}}),
        encoding="utf-8",
    )

    report, _outputs = make_exp11_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        baseline_runs_dir=runs,
        annotation_bundle_dir=workspace_tmpdir / "annotation",
    )
    row = next(
        row for row in report["rows"]
        if (row["model"], row["arm"], row["split"]) == ("1m", "sft", "test")
    )
    assert row["exp11_status"] == "not-run"
    assert row["examples"] is None
    assert row["additional_tokens"] is None
    assert row["additional_flops"] is None
    assert row["training_wandb"]["run_id"] == "main-run-id"
    assert row["training_wandb"]["sync_status"] == "unsynced"
    assert row["training_wandb"]["url"] is None
    assert row["evaluation_wandb"]["run_id"] is None
    markdown = (workspace_tmpdir / "reports" / "intermediate_results.md").read_text(
        encoding="utf-8"
    )
    assert "## Experiment and W&B provenance" in markdown
    assert "exp11-main-1m-sft-seed18" in markdown


def test_exp11_report_measures_continued_sft_to_grpo_flop_ratio(workspace_tmpdir):
    baseline = _snapshot(workspace_tmpdir / "baseline.json")
    runs = workspace_tmpdir / "runs"
    grpo_metrics = runs / "exp11_reasoning_500k_100k" / "1m" / "grpo" / "seed18" / "metrics.csv"
    control_metrics = (
        runs / "exp11_reasoning_500k_100k" / "1m" / "continued_sft" / "seed18" / "metrics.csv"
    )
    sft_metrics = runs / "exp11_reasoning_500k_100k" / "1m" / "sft" / "seed18" / "metrics.csv"
    grpo_metrics.parent.mkdir(parents=True)
    control_metrics.parent.mkdir(parents=True)
    sft_metrics.parent.mkdir(parents=True)
    grpo_metrics.write_text("update,estimated_flops\n100000,8000000000000000\n", encoding="utf-8")
    control_metrics.write_text(
        "optimizer_step,estimated_flops,additional_estimated_flops\n"
        "100000,2800000000000000,2000000000000000\n",
        encoding="utf-8",
    )
    sft_metrics.write_text("optimizer_step,estimated_flops\n100000,800000000000000\n", encoding="utf-8")

    report, _outputs = make_exp11_report(
        runs_dir=runs,
        output_dir=workspace_tmpdir / "reports",
        baseline_snapshot=baseline,
        baseline_runs_dir=runs,
        annotation_bundle_dir=workspace_tmpdir / "annotation",
    )

    matching = report["compute_matching"]["1m"]
    assert matching["status"] == "measured"
    assert matching["grpo_step"] == "100000"
    assert matching["continued_sft_step"] == "100000"
    assert matching["continued_sft_to_grpo_flop_ratio"] == 0.25
    markdown = (workspace_tmpdir / "reports" / "intermediate_results.md").read_text(encoding="utf-8")
    assert "0.2500×" in markdown
