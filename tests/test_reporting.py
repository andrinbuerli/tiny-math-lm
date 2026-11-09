from __future__ import annotations

import json

from tiny_math_lm.reporting import _accuracy_label, collect_results, make_report


def test_accuracy_label_matches_greedy_sampled_or_mixed_results():
    greedy = [{"split": "test", "accuracy": 0.5, "decoding_strategy": "greedy"}]
    sampled = [{"split": "test", "accuracy": 0.5, "decoding_strategy": "sampled"}]

    assert _accuracy_label(greedy) == "Greedy exact accuracy"
    assert _accuracy_label(sampled) == "Sampled exact accuracy"
    assert _accuracy_label(greedy + sampled) == "Exact accuracy (mixed decoding)"


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
    assert "SFT uses 6PN" in comparison["compute_estimate"]
    assert comparison["report_path"] == str(output / "report.md")
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "Evaluation coverage: 0/27" in markdown
    assert "Missing" in markdown
    assert "four-way gradient accumulation" in markdown
    assert "10 examples per split" in markdown
    assert "not full-split accuracy" in markdown
    assert any("fixed qualitative probe" in item for item in comparison["limitations"])


def test_report_documents_dataset_revision_and_limited_controlled_ood_coverage(workspace_tmpdir):
    runs = workspace_tmpdir / "runs"
    output = workspace_tmpdir / "reports"
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "d0rj/mathematics_dataset",
                "dataset_revision": "abc123",
                "split_manifest_sha256": "manifest-hash",
                "tokenizer": {"sha256": "tokenizer-hash"},
                "selected_modules": ["algebra__linear_1d", "arithmetic__mixed"],
                "source_category_counts": {
                    "algebra__linear_1d": {"extrapolate": 0},
                    "arithmetic__mixed": {"extrapolate": 0},
                },
                "selected_pool_category_counts": {
                    "number_length_ood": {
                        "algebra__linear_1d": {"train-hard": 10},
                        "arithmetic__mixed": {"train-hard": 10},
                    },
                    "expression_length_ood": {
                        "algebra__linear_1d": {"train-hard": 5},
                        "arithmetic__mixed": {"train-hard": 5},
                    },
                },
                "split_rules": {
                    "ood_source_by_module": {
                        "algebra__linear_1d": "train categories (controlled fallback; no eligible official extrapolate rows)",
                        "arithmetic__mixed": "train categories (controlled fallback; no eligible official extrapolate rows)",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report, _outputs = make_report(
        runs_dir=runs,
        experiment="pilot",
        output_dir=output,
        seed=17,
        data_bundle_dir=bundle,
    )

    assert report["data_provenance"]["dataset_revision"] == "abc123"
    assert report["data_provenance"]["ood_module_counts"]["expression_length_ood"] == {
        "algebra__linear_1d": 5,
        "arithmetic__mixed": 5,
    }
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "No official `extrapolate` examples were present" in markdown
    assert "algebra__linear_1d: 5; arithmetic__mixed: 5" in markdown


def test_comparison_markdown_summarizes_measured_gains_and_grpo_groups(workspace_tmpdir):
    runs = workspace_tmpdir / "runs"
    output = workspace_tmpdir / "reports"
    seed = 17
    for arm, accuracy in (("sft", 0.4), ("grpo", 0.5), ("continued_sft", 0.45)):
        result_path = (
            runs / "pilot" / "1m" / "evaluation" / arm / "best" / f"seed{seed}" / "results.json"
        )
        result_path.parent.mkdir(parents=True)
        splits = {"test": {"accuracy": accuracy, "examples": 100}}
        if arm == "sft":
            splits["expression_length_ood"] = {
                "accuracy": 0.25,
                "examples": 2000,
                "module": {
                    "algebra__linear_1d": {"accuracy": 0.3},
                    "arithmetic__mixed": {"accuracy": 0.2},
                },
            }
        if arm == "continued_sft":
            splits["number_length_ood"] = {
                "accuracy": 0.0,
                "examples": 0,
                "module": {},
            }
        result_path.write_text(
            json.dumps(
                {
                    "model_parameters": 992640,
                    "checkpoint_step": 1,
                    "decoding": {
                        "strategy": "greedy",
                        "temperature": 0.0,
                        "max_new_tokens": 128,
                        "context_length": 256,
                    },
                    "splits": splits,
                }
            ),
            encoding="utf-8",
        )
        metrics_path = runs / "pilot" / "1m" / arm / f"seed{seed}" / "metrics.csv"
        metrics_path.parent.mkdir(parents=True)
        if arm == "grpo":
            metrics_path.write_text(
                "update,reward/mean,group/mixed_rate,group/all_correct_rate,"
                "group/all_wrong_rate,tokens_seen,estimated_flops\n"
                "1,0.5,0.75,0.125,0.125,200,1000000000000\n"
                "2,0.6,0.25,0.375,0.375,400,2000000000000\n",
                encoding="utf-8",
            )
        else:
            metrics_path.write_text(
                "optimizer_step,tokens_seen,estimated_flops\n"
                "1,500000,500000000000\n"
                "2,1000000,1000000000000\n",
                encoding="utf-8",
            )

    report, _outputs = make_report(
        runs_dir=runs, experiment="pilot", output_dir=output, seed=seed
    )
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "+10.0 pp" in markdown
    assert "+5.0 pp" in markdown
    assert "50.0%" in markdown
    assert "greedy decoding at temperature 0.0; max_new_tokens cap 128; context length 256" in markdown
    assert "| 1m | sft | 40.0% | 500,000 | 500,000,000,000 |" in markdown
    assert "| 1m | grpo | 50.0% | 200 | 1,000,000,000,000 |" in markdown
    assert "2,000 examples / algebra__linear_1d;arithmetic__mixed" in markdown
    assert not report["coverage"]["missing"]["1m"]["sft"]["expression_length_ood"]
    assert report["coverage"]["missing"]["1m"]["continued_sft"]["number_length_ood"]
    measured_test = next(
        row
        for row in report["available_measurements"]
        if row["arm"] == "sft" and row["split"] == "test"
    )
    assert measured_test["decoding_strategy"] == "greedy"
    assert measured_test["max_new_tokens"] == 128
    csv_header = (output / "comparison.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "max_new_tokens" in csv_header
    empty_split = next(
        row
        for row in report["available_measurements"]
        if row["arm"] == "continued_sft" and row["split"] == "number_length_ood"
    )
    assert empty_split["accuracy"] is None
    ood_measurement = next(
        row
        for row in report["available_measurements"]
        if row["arm"] == "sft" and row["split"] == "expression_length_ood"
    )
    assert ood_measurement["module_count"] == 2
    assert "SFT token counts include processed input positions" in markdown


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
