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
