from pathlib import Path

from tiny_math_lm.eval.exp16_report import build_exp16_report, paired_bootstrap


def _record(module, question, target, correct):
    return {
        "module": module,
        "question": question,
        "target": target,
        "correct": correct,
    }


def test_paired_bootstrap_is_question_aligned_and_deterministic():
    sft = [
        _record("arithmetic__mul", "q1", "1", "true"),
        _record("arithmetic__mul", "q2", "2", "false"),
        _record("arithmetic__div", "q3", "3", "false"),
    ]
    grpo = [
        _record("arithmetic__mul", "q1", "1", "true"),
        _record("arithmetic__mul", "q2", "2", "true"),
        _record("arithmetic__div", "q3", "3", "false"),
        _record("arithmetic__div", "extra", "4", "true"),
    ]
    first = paired_bootstrap(sft, grpo, seed=18, replicates=200)
    second = paired_bootstrap(sft, grpo, seed=18, replicates=200)
    assert first == second
    assert first["questions"] == 3
    assert first["accuracy_difference"] == 1 / 3
    assert first["uncertainty_scope"] == "within-question sampling; not between-seed uncertainty"


def test_report_marks_missing_arms_without_fabricating_measurements(tmp_path):
    output = tmp_path / "report"
    report = build_exp16_report(
        runs_root=tmp_path / "runs",
        output_dir=output,
        seed=18,
    )
    assert report["arms_present"] == []
    assert set(report["arms_missing"]) == {
        "base_zero_shot", "sft", "grpo", "continued_sft",
    }
    assert report["paired_comparisons"] == []
    assert (output / "results.csv").is_file()
    assert (output / "results.json").is_file()
    assert "Pretrained-data contamination cannot be excluded." in (output / "report.md").read_text()


def test_report_outputs_stay_inside_requested_namespace(tmp_path):
    output = tmp_path / "reports" / "exp16_qwen3_base_reasoning_v3" / "seed18"
    report = build_exp16_report(runs_root=tmp_path / "runs", output_dir=output, seed=18)
    assert Path(output / "report.md").is_file()
    assert report["limitations"]
