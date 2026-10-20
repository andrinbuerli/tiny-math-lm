"""Paired question-level comparison artifacts for Experiment 16."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("base_zero_shot", "sft", "grpo", "continued_sft")
SPLITS = ("test", "number_length_ood", "expression_length_ood")
FORMATS = ("ordinary_qa", "reasoning_executor")


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap paired accuracy differences over fixed question identities."""
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return row["module"], row["question"], row["target"]

    left_map = {key(row): row for row in left}
    right_map = {key(row): row for row in right}
    common = sorted(left_map.keys() & right_map.keys())
    if not common:
        raise ValueError("Paired evaluation files have no shared question identities")
    differences = np.asarray(
        [int(_bool(right_map[item]["correct"])) - int(_bool(left_map[item]["correct"])) for item in common],
        dtype=np.float64,
    )
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    counts = np.asarray([(differences == value).sum() for value in (-1, 0, 1)])
    draws = rng.multinomial(len(differences), counts / len(differences), size=replicates)
    estimates = (draws[:, 2] - draws[:, 0]) / len(differences)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return {
        "questions": len(common),
        "accuracy_difference": observed,
        "ci_95_low": float(lower),
        "ci_95_high": float(upper),
        "method": "paired question-level percentile bootstrap",
        "replicates": replicates,
        "seed": seed,
        "uncertainty_scope": "within-question sampling; not between-seed uncertainty",
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_arm(root: Path, arm: str, seed: int) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    selector = {
        "base_zero_shot": ("base_zero_shot", "pristine"),
        "sft": ("sft", "best"),
        "grpo": ("grpo", "final"),
        "continued_sft": ("continued_sft", "final"),
    }
    checkpoint_arm, checkpoint_id = selector[arm]
    directory = root / "evaluation" / checkpoint_arm / checkpoint_id / f"seed{seed}"
    result_path = directory / "results.json"
    if not result_path.is_file():
        return None, {}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = {}
    for split in SPLITS:
        for format_name in FORMATS:
            path = directory / f"{split}_{format_name}.csv"
            if path.is_file():
                predictions[f"{split}/{format_name}"] = _rows(path)
    return result, predictions


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "arm", "split", "format", "level", "category", "metric", "value",
        "comparison", "paired_questions", "ci_95_low", "ci_95_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_training_csv(path: Path, histories: dict[str, list[dict[str, Any]]]) -> None:
    fields = sorted({"arm", *(key for rows in histories.values() for row in rows for key in row)})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm, rows in histories.items():
            writer.writerows({"arm": arm, **row} for row in rows)


def _figures(
    output_dir: Path,
    summaries: dict[str, dict[str, Any]],
    training: dict[str, list[dict[str, Any]]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    available = [arm for arm in ARMS if arm in summaries]
    if available:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, format_name in zip(axes, FORMATS, strict=True):
            for arm in available:
                values = [
                    summaries[arm].get("splits", {}).get(f"{split}/{format_name}", {}).get("accuracy")
                    for split in SPLITS
                ]
                axis.plot(SPLITS, values, marker="o", label=arm)
            axis.set_title(format_name.replace("_", " "))
            axis.set_ylim(0, 1)
            axis.set_ylabel("Mathematical accuracy")
            axis.tick_params(axis="x", rotation=20)
            axis.grid(alpha=0.25)
        axes[-1].legend(loc="best")
        figure.tight_layout()
        for extension in ("png", "svg"):
            path = output_dir / f"accuracy_by_arm.{extension}"
            figure.savefig(path, dpi=160)
            paths.append(str(path))
        plt.close(figure)

        for mode in FORMATS:
            modules = sorted(
                {
                    module
                    for arm in available
                    for module in summaries[arm].get("splits", {}).get(f"test/{mode}", {}).get("by_module", {})
                }
            )
            if not modules:
                continue
            figure, axis = plt.subplots(figsize=(13, 5))
            x = np.arange(len(modules))
            width = 0.8 / len(available)
            for offset, arm in enumerate(available):
                values = [
                    summaries[arm].get("splits", {}).get(f"test/{mode}", {}).get("by_module", {}).get(module, {}).get("accuracy")
                    for module in modules
                ]
                axis.bar(x + offset * width, values, width=width, label=arm)
            axis.set_xticks(x + width * (len(available) - 1) / 2, modules, rotation=25, ha="right")
            axis.set_ylim(0, 1)
            axis.set_ylabel("Test mathematical accuracy")
            axis.set_title(mode.replace("_", " "))
            axis.legend()
            figure.tight_layout()
            for extension in ("png", "svg"):
                path = output_dir / f"test_accuracy_by_module_{mode}.{extension}"
                figure.savefig(path, dpi=160)
                paths.append(str(path))
            plt.close(figure)

    for arm, rows in training.items():
        if not rows:
            continue
        fields = (
            ("optimizer_step", "train/completion_loss", "validation/completion_loss")
            if arm in {"sft", "continued_sft"}
            else ("update", "reward/mean", "entropy", "kl", "group/mixed_rate")
        )
        step_field = fields[0]
        figure, axis = plt.subplots(figsize=(8, 4))
        for metric in fields[1:]:
            x_values = []
            y_values = []
            for row in rows:
                try:
                    x_values.append(float(row.get(step_field)))
                    y_values.append(float(row.get(metric)))
                except (TypeError, ValueError):
                    continue
            if x_values:
                axis.plot(x_values, y_values, label=metric)
        axis.set_xlabel(step_field.replace("_", " "))
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        for extension in ("png", "svg"):
            path = output_dir / f"{arm}_training_curves.{extension}"
            figure.savefig(path, dpi=160)
            paths.append(str(path))
        plt.close(figure)
    return paths


def build_exp16_report(*, runs_root: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    """Aggregate only existing Experiment 16 artifacts and label missing arms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = runs_root / "exp16_qwen3_base_reasoning_v3" / "qwen3_0_6b_base"
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing = []
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        summary, arm_predictions = _load_arm(run_root, arm, seed)
        if summary is None:
            missing.append(arm)
            continue
        results[arm] = summary
        predictions[arm] = arm_predictions
        for split_mode, values in summary.get("splits", {}).items():
            split, format_name = split_mode.split("/", maxsplit=1)
            for metric, value in values.items():
                if metric == "by_module":
                    for module, metrics in value.items():
                        for name, score in metrics.items():
                            if isinstance(score, (int, float)) and score is not None:
                                rows.append({
                                    "arm": arm, "split": split, "format": format_name,
                                    "level": "module", "category": module, "metric": name, "value": score,
                                })
                elif metric in {"by_digit_width", "by_operator_count"}:
                    for category, metrics in value.items():
                        for name, score in metrics.items():
                            if isinstance(score, (int, float)) and score is not None:
                                rows.append({
                                    "arm": arm, "split": split, "format": format_name,
                                    "level": metric.removeprefix("by_"), "category": category,
                                    "metric": name, "value": score,
                                })
                elif isinstance(value, (int, float)) and value is not None:
                    rows.append({
                        "arm": arm, "split": split, "format": format_name,
                        "level": "overall", "metric": metric, "value": value,
                    })

    comparisons = []
    pairings = (
        ("sft", "base_zero_shot"),
        ("grpo", "base_zero_shot"),
        ("grpo", "sft"),
        ("continued_sft", "base_zero_shot"),
        ("continued_sft", "sft"),
    )
    expected_counts = {"test": 50_000, "number_length_ood": 5_000, "expression_length_ood": 2_000}
    for target_arm, reference_arm in pairings:
        if reference_arm not in predictions or target_arm not in predictions:
            continue
        for split in SPLITS:
            for mode in FORMATS:
                key = f"{split}/{mode}"
                left = predictions[reference_arm].get(key)
                right = predictions[target_arm].get(key)
                if left is None or right is None:
                    continue
                if len(left) != expected_counts[split] or len(right) != expected_counts[split]:
                    raise ValueError(
                        f"Paired bootstrap requires the full fixed {split} evaluation population"
                    )
                interval = paired_bootstrap(left, right, seed=seed)
                if interval["questions"] != expected_counts[split]:
                    raise ValueError(f"Paired rows do not cover the full fixed {split} population")
                comparison = f"{target_arm}-{reference_arm}"
                comparisons.append({
                    "arm": target_arm, "split": split, "format": mode,
                    "comparison": comparison, **interval,
                })
                rows.append({
                    "arm": target_arm, "split": split, "format": mode, "level": "paired",
                    "metric": "accuracy_difference_vs_sft",
                    "value": interval["accuracy_difference"],
                    "comparison": comparison,
                    "paired_questions": interval["questions"],
                    "ci_95_low": interval["ci_95_low"],
                    "ci_95_high": interval["ci_95_high"],
                })

    training_history = {}
    training_summary = {}
    for arm in ("sft", "grpo", "continued_sft"):
        trainer_arm = "continued_sft" if arm == "continued_sft" else arm
        trainer_phase = "continued_sft" if arm == "continued_sft" else "main"
        training_dir = run_root / trainer_arm / trainer_phase / f"seed{seed}"
        path = training_dir / "metrics.csv"
        if path.is_file():
            training_history[arm] = _rows(path)
        summary_path = training_dir / "training_summary.json"
        if summary_path.is_file():
            training_summary[arm] = json.loads(summary_path.read_text(encoding="utf-8"))
        elif arm == "grpo" and (training_dir / "latest.pt").is_file():
            import torch

            state = torch.load(training_dir / "latest.pt", map_location="cpu", weights_only=False)
            last_metrics = training_history.get(arm, [])[-1] if training_history.get(arm) else {}
            training_summary[arm] = {
                "step": state.get("update"),
                "tokens_seen": state.get("tokens_seen"),
                "policy_tokens": state.get("policy_tokens"),
                "executor_tokens": state.get("executor_tokens"),
                "estimated_flops": state.get("estimated_flops"),
                "prompt_examples": state.get("prompt_examples"),
                "wall_seconds": state.get("wall_seconds"),
                "tokens_per_second": (
                    state.get("tokens_seen", 0) / state["wall_seconds"]
                    if state.get("wall_seconds") else None
                ),
                "peak_gpu_memory_bytes": state.get("peak_gpu_memory_bytes"),
                "final_metrics": last_metrics,
            }
    for arm, summary in training_summary.items():
        for metric, value in summary.items():
            if isinstance(value, (int, float)) and value is not None:
                rows.append({
                    "arm": arm, "level": "training", "metric": metric, "value": value,
                })
    _write_training_csv(output_dir / "training_metrics.csv", training_history)
    figure_paths = _figures(output_dir, results, training_history)
    payload = {
        "experiment": "exp16_qwen3_base_reasoning_v3",
        "seed": seed,
        "arms_present": list(results),
        "arms_missing": missing,
        "results": results,
        "paired_comparisons": comparisons,
        "training_summary": training_summary,
        "figures": figure_paths,
        "uncertainty_note": "Paired bootstrap intervals describe question-level sampling uncertainty for one seed; they do not estimate between-seed uncertainty.",
        "limitations": [
            "Pretrained-data contamination cannot be excluded.",
            "Only one seed was run.",
            "Historical TinyMath comparisons differ in tokenizer, architecture, model size, and pretraining.",
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "results.csv", rows)
    missing_text = ", ".join(missing) if missing else "none"
    accuracy_lines = [
        "| Arm | Split | Format | Accuracy | Format validity | Protocol success |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        if arm not in results:
            continue
        for split in SPLITS:
            for format_name in FORMATS:
                values = results[arm].get("splits", {}).get(f"{split}/{format_name}")
                if values:
                    accuracy_lines.append(
                        f"| {arm} | {split} | {format_name} | {values.get('accuracy')} | "
                        f"{values.get('format_validity')} | {values.get('full_protocol_success')} |"
                    )
    interval_lines = [
        "| Comparison | Split | Format | Accuracy difference | 95% paired CI | Questions |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        interval_lines.append(
            f"| {item['comparison']} | {item['split']} | {item['format']} | "
            f"{item['accuracy_difference']:.4f} | [{item['ci_95_low']:.4f}, {item['ci_95_high']:.4f}] | "
            f"{item['questions']} |"
        )
    training_lines = [
        "| Arm | Tokens | Estimated FLOPs | Wall seconds | Tokens/s | Peak GPU bytes | Target FLOPs | Overshoot FLOPs | Match error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("sft", "grpo", "continued_sft"):
        summary = training_summary.get(arm)
        if not summary:
            continue
        estimated = summary.get("estimated_flops", summary.get("additional_estimated_flops"))
        training_lines.append(
            f"| {arm} | {summary.get('tokens_seen')} | {estimated} | {summary.get('wall_seconds')} | "
            f"{summary.get('tokens_per_second')} | {summary.get('peak_gpu_memory_bytes')} | "
            f"{summary.get('compute_budget_flops')} | {summary.get('compute_overshoot_flops')} | "
            f"{summary.get('compute_matching_error')} |"
        )
    markdown = [
        "# Experiment 16 results",
        "",
        f"Arms with evaluation artifacts: {', '.join(results) if results else 'none'}.",
        f"Arms without evaluation artifacts: {missing_text}.",
        "",
        "## Evaluation metrics",
        "",
        *accuracy_lines,
        "",
        "## Paired comparisons",
        "",
        *interval_lines,
        "",
        "CSV and JSON include per-module, digit-width, operator-count and training diagnostics. Figures are exported as PNG and SVG.",
        "",
        "## Training and compute",
        "",
        *training_lines,
        "",
        "The GRPO estimator counts rollout forward FLOPs as 2PN plus policy forward/backward FLOPs as 6PN. Continued SFT uses 6PN, with N equal to nonpadding input tokens and P the loaded parameter count. The estimate excludes optimizer and KV-cache overhead, executor arithmetic, and hardware utilization effects.",
        "",
        "Paired intervals quantify within-question sampling uncertainty for this fixed seed, not between-seed uncertainty.",
        "",
        "Pretrained-data contamination cannot be excluded. Only one seed was run. Historical TinyMath comparisons differ in tokenizer, architecture, model size, and pretraining.",
        "",
        "Missing stages are left missing; no measurements are imputed.",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return payload
