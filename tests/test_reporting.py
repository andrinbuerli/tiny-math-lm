from __future__ import annotations

import json

from tinymath.reporting import collect_results, make_report


def test_comparison_report_marks_missing_runs_without_fabricating_values(workspace_tmpdir):
    runs = workspace_tmpdir / "runs"
    output = workspace_tmpdir / "reports"
    report, outputs = make_report(runs_dir=runs, experiment="pilot", output_dir=output, seed=17)
    assert report["coverage"]["available_evaluations"] == 0
    assert all(report["coverage"]["missing"][model][arm]["test"] for model in ("1m", "10m", "100m")
               for arm in ("sft", "grpo", "continued_sft"))
    assert all(path.exists() for path in outputs)
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 9
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["available_measurements"] == []
    assert "6 x actual trainable parameters" in comparison["compute_estimate"]


def test_result_collection_keeps_one_seed_coherent(workspace_tmpdir):
    runs = workspace_tmpdir / "runs"
    for seed, accuracy in ((17, 0.25), (18, 0.75)):
        path = runs / "pilot" / "1m" / "evaluation" / "sft" / "best" / f"seed{seed}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "model_parameters": 992640,
                    "splits": {"test": {"accuracy": accuracy, "examples": 4}},
                }
            ),
            encoding="utf-8",
        )
    rows, coverage = collect_results(runs, "pilot", seed=17)
    assert len(rows) == 1
    assert rows[0]["accuracy"] == 0.25
    assert coverage["available_evaluations"] == 1
