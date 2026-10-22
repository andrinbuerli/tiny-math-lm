from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "update_exp16_intermediate", Path(__file__).parents[1] / "scripts/update_exp16_intermediate.py"
)
assert SPEC is not None and SPEC.loader is not None
REPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)
_read_jsonl = REPORTER._read_jsonl
_render_markdown = REPORTER._render_markdown
_summarize = REPORTER._summarize
_training_results = REPORTER._training_results


def test_summary_reports_rates_and_diagnostics() -> None:
    rows = [
        {
            "correct": True,
            "format_valid": True,
            "full_success": True,
            "trace_valid": True,
            "capped": False,
            "executor_failure": False,
            "output_tokens": 7,
            "call_count": 1,
            "policy_tokens": 5,
            "executor_tokens": 2,
            "module": "arithmetic__add_or_sub",
            "digit_width": 2,
            "operator_count": 1,
        },
        {
            "correct": False,
            "format_valid": False,
            "full_success": False,
            "trace_valid": False,
            "capped": True,
            "executor_failure": True,
            "output_tokens": 9,
            "call_count": 0,
            "policy_tokens": 9,
            "executor_tokens": 0,
            "module": "arithmetic__add_or_sub",
            "digit_width": 3,
            "operator_count": 2,
        },
    ]

    summary = _summarize(rows)

    assert summary["rows"] == 2
    assert summary["correct"] == 0.5
    assert summary["format_valid"] == 0.5
    assert summary["mean_output_tokens"] == 8
    assert summary["executor_calls"] == 1
    assert summary["policy_tokens"] == 14
    assert summary["executor_tokens"] == 2
    assert summary["modules"]["arithmetic__add_or_sub"]["accuracy"] == 0.5
    assert list(summary["by_digit_width"]) == ["2", "3"]
    assert list(summary["by_operator_count"]) == ["1", "2"]
    assert summary["by_digit_width"]["2"]["accuracy"] == 1.0
    assert summary["by_operator_count"]["2"]["accuracy"] == 0.0


def test_jsonl_reader_ignores_partial_trailing_record(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_text('{"correct": true}\n{"correct":', encoding="utf-8")

    assert _read_jsonl(path) == [{"correct": True}]


def test_markdown_labels_partial_results_and_decodes_split_format() -> None:
    report = _render_markdown(
        {
            "evaluation/base_zero_shot/pristine/seed18/progress/number_length_ood_ordinary_qa.jsonl": {
                "rows": 2,
                "correct": 0.5,
                "format_valid": 1.0,
                "full_success": 0.0,
                "trace_valid": 0.0,
                "capped": 0.0,
                "mean_output_tokens": 3.0,
                "executor_calls": 0,
                "executor_failure": 0.0,
                "policy_tokens": 6,
                "executor_tokens": 0,
                "modules": {},
            }
        },
        "2026-10-21T11:00:00+00:00",
    )

    assert "| base_zero_shot/pristine | number_length_ood | ordinary_qa | 2 | 5000 |" in report
    assert "Live, incomplete results" in report


def test_markdown_shows_qualitative_validation_rows_without_promoting_them_to_metrics():
    report = _render_markdown(
        {},
        "now",
        qualitative_results=[
            {
                "optimizer_step": 5000,
                "module": "arithmetic__add_or_sub",
                "question": "2 + 2?",
                "target": "4",
                "prediction": "4",
                "format_valid": True,
                "trace_valid": True,
                "correct": True,
                "full_success": True,
                "completion": "<think>\\n2 + 2 = 4\\n</think>\\nA: 4",
            }
        ],
    )

    assert "## Qualitative validation examples" in report
    assert "illustrative examples, not an accuracy estimate" in report
    assert "2 + 2?" in report
    assert "Generated completion:" in report


def test_markdown_includes_tracked_pilot_metrics_and_incomplete_status() -> None:
    report = _render_markdown(
        {},
        "2026-10-21T13:00:00+00:00",
        [
            {
                "arm": "GRPO",
                "phase": "optimization_pilot",
                "status": "finished",
                "sync_status": "synced",
                "updates": "1",
                "wall_seconds": "129.37",
                "tokens_per_second": "538.38",
                "peak_gpu_memory_bytes": "12289678848",
                "wandb_id": "pilot123",
                "wandb_url": "https://wandb.ai/example/pilot123",
                "has_metrics": True,
            },
            {
                "arm": "GRPO",
                "phase": "smoke",
                "status": "failed",
                "sync_status": "synced",
                "updates": None,
                "wall_seconds": None,
                "tokens_per_second": None,
                "peak_gpu_memory_bytes": None,
                "wandb_id": "failed123",
                "wandb_url": "https://wandb.ai/example/failed123",
                "has_metrics": False,
            },
        ],
    )

    assert "| GRPO | optimization_pilot | finished | 1 | 129.37 | 538.38 | 12.29 |" in report
    assert "[pilot123](https://wandb.ai/example/pilot123)" in report
    assert (
        "Failed or interrupted pilots without a completed optimizer update have no metrics row"
        in report
    )


def test_training_results_read_latest_main_metrics_and_mark_run_live(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / REPORTER.EXPERIMENT_DIR
        / REPORTER.MODEL_DIR
        / "sft"
        / "main"
        / "seed18"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text(
        "optimizer_step,epoch,train/completion_loss,validation/completion_loss,"
        "validation/greedy_accuracy,validation/greedy_accuracy_examples,tokens_seen,"
        "tokens_per_second,wall_seconds,peak_gpu_memory_bytes\n"
        "1,0.1,2.5,,,,1024,900,1.2,5000000000\n"
        "2,0.2,2.1,2.3,0.375,8,2048,1000,2.4,6000000000\n"
        "3,0.3,1.9,,, ,3072,1100,3.6,7000000000\n",
        encoding="utf-8",
    )

    results = _training_results(tmp_path, 18)
    assert len(results) == 1
    assert results[0]["status"] == "running"
    assert results[0]["latest"]["optimizer_step"] == "3"
    assert results[0]["latest"]["validation/completion_loss"] == "2.3"
    assert results[0]["latest"]["validation_step"] == "2"
    assert results[0]["latest"]["validation/greedy_accuracy"] == "0.375"
    assert results[0]["latest"]["greedy_accuracy_step"] == "2"
    assert len(results[0]["metrics"]) == 3

    report = _render_markdown({}, "now", training_results=results)
    assert "## Main training progress" in report
    assert "| SFT | main | running | 3 | 0.300 | 1.900 | 2.300 @ 2 | 37.50% (8, 2) | 3072 |" in report


def test_refresh_updates_training_report_and_all_plot_formats(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    output_dir = tmp_path / "reports"
    model_root = run_root / REPORTER.EXPERIMENT_DIR / REPORTER.MODEL_DIR
    metrics_dir = model_root / "sft" / "main" / "seed18"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.csv").write_text(
        "optimizer_step,epoch,train/completion_loss,validation/completion_loss,"
        "tokens_seen,tokens_per_second,wall_seconds,peak_gpu_memory_bytes\n"
        "1,0.1,2.5,2.4,1024,900,1.2,5000000000\n",
        encoding="utf-8",
    )
    progress_dir = model_root / "evaluation" / "base_zero_shot" / "pristine" / "seed18" / "progress"
    progress_dir.mkdir(parents=True)
    (progress_dir / "test_ordinary_qa.jsonl").write_text(
        '{"correct": true, "digit_width": 1, "operator_count": 1}\n',
        encoding="utf-8",
    )

    REPORTER.refresh(runs_root=run_root, output_dir=output_dir, seed=18)

    report = (output_dir / "intermediate_results.md").read_text(encoding="utf-8")
    assert "| SFT | main | running | 1 |" in report
    assert "base_zero_shot/pristine" in report
    for stem in (
        "intermediate_accuracy",
        "intermediate_progress",
        "intermediate_by_digit_width",
        "intermediate_by_operator_count",
        "intermediate_training_loss",
    ):
        assert (output_dir / f"{stem}.png").is_file()
        assert (output_dir / f"{stem}.svg").is_file()
