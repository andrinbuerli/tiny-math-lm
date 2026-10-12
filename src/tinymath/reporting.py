"""Collect measured local runs and render comparable figures and tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ("1m", "10m", "100m")
ARMS = ("sft", "grpo", "continued_sft")
SPLITS = ("test", "number_length_ood", "expression_length_ood")


def collect_results(
    runs_dir: Path, experiment: str, seed: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = runs_dir / experiment
    rows: list[dict[str, Any]] = []
    by_checkpoint: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*/evaluation/*/*/seed*/results.json")):
        if seed is not None and path.parent.name != f"seed{seed}":
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        model_name = path.parents[4].name
        arm = path.parents[2].name
        for split_name, split in summary.get("splits", {}).items():
            row = {
                "model": model_name,
                "arm": arm,
                "split": split_name,
                "parameters": summary.get("model_parameters"),
                "accuracy": split.get("accuracy"),
                "format_validity": split.get("format_validity"),
                "mean_output_tokens": split.get("mean_output_tokens"),
                "examples": split.get("examples"),
                "checkpoint": summary.get("checkpoint"),
            }
            rows.append(row)
            by_checkpoint[(model_name, arm)] = summary
    missing = {
        model: {arm: {split: True for split in SPLITS} for arm in ARMS}
        for model in MODELS
    }
    for row in rows:
        if row["model"] in missing and row["arm"] in missing[row["model"]]:
            missing[row["model"]][row["arm"]][row["split"]] = False
    return rows, {"available_evaluations": len(by_checkpoint), "missing": missing}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    outputs = [output_dir / f"{name}.{ext}" for ext in ("png", "svg")]
    for path in outputs:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outputs


def _no_data(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.text(0.5, 0.5, "No completed measurements", ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def _figure_scaling(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    available = [row for row in rows if row["split"] == "test" and row["parameters"] and row["accuracy"] is not None]
    for arm in ARMS:
        values = sorted((row for row in available if row["arm"] == arm), key=lambda row: row["parameters"])
        if values:
            axis.plot([row["parameters"] for row in values], [row["accuracy"] for row in values], marker="o", label=arm)
    if available:
        axis.set_xscale("log")
        axis.set_xlabel("Actual trainable parameters")
        axis.set_ylabel("Greedy exact accuracy")
        axis.set_title("Model scaling on the fixed ID test set")
        axis.legend()
        axis.grid(alpha=0.25)
    else:
        _no_data(axis, "Model scaling")
    return _save(fig, output_dir, "model_scaling")


def _figure_arm_bars(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    values = {(row["model"], row["arm"]): row["accuracy"] for row in rows if row["split"] == "test"}
    found = any(value is not None for value in values.values())
    if found:
        width = 0.24
        positions = list(range(len(MODELS)))
        for index, arm in enumerate(ARMS):
            bar_values = [values.get((model, arm)) for model in MODELS]
            axis.bar([pos + (index - 1) * width for pos in positions],
                     [value if value is not None else 0 for value in bar_values],
                     width, label=arm)
            for pos, value in zip(positions, bar_values, strict=True):
                if value is None:
                    axis.text(pos + (index - 1) * width, 0.01, "missing", ha="center", rotation=90, fontsize=7)
        axis.set_xticks(positions, MODELS)
        axis.set_ylabel("Greedy exact accuracy")
        axis.set_title("SFT, GRPO and continued-SFT controls")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    else:
        _no_data(axis, "Training arm comparison")
    return _save(fig, output_dir, "training_arm_accuracy")


def _figure_modules(
    rows: list[dict[str, Any]], output_dir: Path, runs_dir: Path, experiment: str, seed: int | None
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(9, 5))
    result_paths = sorted((runs_dir / experiment).glob("*/evaluation/*/*/seed*/results.json"))
    modules = {}
    for path in result_paths:
        if seed is not None and path.parent.name != f"seed{seed}":
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        model, arm = path.parents[4].name, path.parents[2].name
        for module, metrics in summary.get("splits", {}).get("test", {}).get("module", {}).items():
            modules[(model, arm, module)] = metrics["accuracy"]
    names = sorted({key[2] for key in modules})
    if names:
        labels = [name.split("__")[-1] for name in names]
        width = 0.8 / max(1, len(MODELS) * len(ARMS))
        index = 0
        for model in MODELS:
            for arm in ARMS:
                values = [modules.get((model, arm, module)) for module in names]
                if any(value is not None for value in values):
                    offset = (index - (len(MODELS) * len(ARMS) - 1) / 2) * width
                    axis.bar([i + offset for i in range(len(names))],
                             [value or 0 for value in values], width, label=f"{model} {arm}")
                index += 1
        axis.set_xticks(range(len(names)), labels, rotation=25, ha="right")
        axis.set_ylabel("Greedy exact accuracy")
        axis.set_title("Per-module ID test accuracy")
        axis.legend(fontsize=7, ncol=2)
    else:
        _no_data(axis, "Per-module breakdown")
    return _save(fig, output_dir, "per_module_accuracy")


def _figure_ood(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    wanted = {(row["model"], row["arm"], row["split"]): row["accuracy"] for row in rows}
    present = any(value is not None for value in wanted.values())
    if present:
        categories = ("test", "number_length_ood", "expression_length_ood")
        width = 0.8 / len(categories)
        x = list(range(len(MODELS) * len(ARMS)))
        labels = [f"{model}\n{arm}" for model in MODELS for arm in ARMS]
        for index, split in enumerate(categories):
            vals = [wanted.get((model, arm, split)) for model in MODELS for arm in ARMS]
            axis.bar([pos + (index - 1) * width for pos in x], [v or 0 for v in vals], width, label=split)
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_ylabel("Greedy exact accuracy")
        axis.set_title("In-distribution and controlled OOD accuracy")
        axis.legend(fontsize=8)
    else:
        _no_data(axis, "ID and OOD")
    return _save(fig, output_dir, "id_vs_ood")


def _figure_digits(
    rows: list[dict[str, Any]], output_dir: Path, runs_dir: Path, experiment: str, seed: int | None
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    found = False
    for path in sorted((runs_dir / experiment).glob("*/evaluation/*/*/seed*/results.json")):
        if seed is not None and path.parent.name != f"seed{seed}":
            continue
        if path.parents[2].name != "sft":
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        curve = summary.get("splits", {}).get("test", {}).get("by_digit_width", {})
        if curve:
            found = True
            axis.plot([int(key) for key in curve], [value["accuracy"] for key, value in sorted(curve.items(), key=lambda item: int(item[0]))],
                      marker="o", label=path.parents[4].name)
    if found:
        axis.set_xlabel("Longest decimal operand width in question")
        axis.set_ylabel("Greedy exact accuracy")
        axis.set_title("SFT accuracy by operand digit width")
        axis.legend()
        axis.grid(alpha=0.25)
    else:
        _no_data(axis, "Accuracy by digit width")
    return _save(fig, output_dir, "accuracy_by_digit_width")


def _figure_operators(
    rows: list[dict[str, Any]], output_dir: Path, runs_dir: Path, experiment: str, seed: int | None
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    found = False
    for path in sorted((runs_dir / experiment).glob("*/evaluation/*/*/seed*/results.json")):
        if seed is not None and path.parent.name != f"seed{seed}":
            continue
        if path.parents[2].name != "sft":
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        curve = summary.get("splits", {}).get("test", {}).get("by_operator_count", {})
        if curve:
            found = True
            ordered = sorted(curve.items(), key=lambda item: int(item[0]))
            axis.plot([int(key) for key, _value in ordered], [value["accuracy"] for _key, value in ordered],
                      marker="o", label=path.parents[4].name)
    if found:
        axis.set_xlabel("Operator count in question")
        axis.set_ylabel("Greedy exact accuracy")
        axis.set_title("SFT accuracy by expression length")
        axis.legend()
        axis.grid(alpha=0.25)
    else:
        _no_data(axis, "Accuracy by expression length")
    return _save(fig, output_dir, "accuracy_by_expression_length")


def _figure_curves(runs_dir: Path, experiment: str, output_dir: Path, seed: int | None) -> list[Path]:
    outputs: list[Path] = []
    for figure_name, columns, title in (
        ("training_loss", ("train/loss", "val/loss"), "SFT training and validation loss"),
        ("reward_entropy_kl", ("reward/mean", "entropy", "kl"), "GRPO reward, entropy and KL"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        found = False
        for path in sorted((runs_dir / experiment).glob("*/*/*/seed*/metrics.csv")):
            if seed is not None and path.parent.name != f"seed{seed}":
                continue
            arm = path.parent.parent.name
            if figure_name == "training_loss" and arm not in ("sft", "continued_sft"):
                continue
            if figure_name == "reward_entropy_kl" and arm != "grpo":
                continue
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for column in columns:
                values = [(int(row.get("optimizer_step", row.get("update", 0))), float(row[column]))
                          for row in rows if row.get(column) not in (None, "")]
                if values:
                    found = True
                    axis.plot([item[0] for item in values], [item[1] for item in values],
                              label=f"{path.parents[2].name} {arm} {column}")
        if found:
            axis.set_xlabel("Step")
            axis.set_title(title)
            axis.legend(fontsize=7)
            axis.grid(alpha=0.25)
        else:
            _no_data(axis, title)
        outputs.extend(_save(fig, output_dir, figure_name))
    return outputs


def _figure_compute(
    rows: list[dict[str, Any]], runs_dir: Path, experiment: str, output_dir: Path, seed: int | None
) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    points: dict[tuple[str, str], tuple[float, float]] = {}
    for path in sorted((runs_dir / experiment).glob("*/*/*/seed*/metrics.csv")):
        if seed is not None and path.parent.name != f"seed{seed}":
            continue
        arm = path.parent.parent.name
        with path.open(encoding="utf-8") as handle:
            curve = list(csv.DictReader(handle))
        if not curve:
            continue
        token_key = "tokens_seen"
        if token_key not in curve[-1]:
            continue
        result = next((row for row in rows if row["model"] == path.parents[2].name and row["arm"] == arm), None)
        if result and result["parameters"]:
            estimate = curve[-1].get("estimated_flops")
            flops = (
                int(float(estimate))
                if estimate not in (None, "")
                else 6 * result["parameters"] * int(float(curve[-1][token_key]))
            )
            points[(path.parents[2].name, arm)] = (flops, 0)
    for row in rows:
        key = (row["model"], row["arm"])
        if key in points and row["split"] == "test" and row["accuracy"] is not None:
            compute, _ = points[key]
            axis.scatter(compute, row["accuracy"], label=f"{row['model']} {row['arm']}")
    if points:
        axis.set_xscale("log")
        axis.set_xlabel("Estimated training FLOPs (dense 2N/6N token-position model)")
        axis.set_ylabel("Greedy ID test accuracy")
        axis.set_title("Accuracy versus estimated training compute")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
    else:
        _no_data(axis, "Accuracy versus compute")
    return _save(fig, output_dir, "accuracy_vs_compute")


def make_report(
    *, runs_dir: Path, experiment: str, output_dir: Path, seed: int | None = None
) -> tuple[dict[str, Any], list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, coverage = collect_results(runs_dir, experiment, seed)
    expected_rows = []
    actual = {(row["model"], row["arm"], row["split"]): row for row in rows}
    for model in MODELS:
        for arm in ARMS:
            for split in SPLITS:
                expected_rows.append(
                    actual.get((model, arm, split), {
                        "model": model, "arm": arm, "split": split, "parameters": "",
                        "accuracy": "", "format_validity": "", "mean_output_tokens": "",
                        "examples": "", "checkpoint": "",
                    })
                )
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "arm", "split", "parameters", "accuracy", "format_validity",
                  "mean_output_tokens", "examples", "checkpoint"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(expected_rows)
    report = {
        "experiment": experiment,
        "seed": seed,
        "available_measurements": rows,
        "coverage": coverage,
        "compute_estimate": "SFT uses 6PN; GRPO counts no-cache rollout prefix positions and policy/reference token positions with dense 2N/6N multipliers",
        "limitations": ["Compute uses a dense-transformer token-position approximation; it is not hardware-counter FLOPs."],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    figures = []
    figures += _figure_scaling(rows, output_dir)
    figures += _figure_arm_bars(rows, output_dir)
    figures += _figure_modules(rows, output_dir, runs_dir, experiment, seed)
    figures += _figure_digits(rows, output_dir, runs_dir, experiment, seed)
    figures += _figure_operators(rows, output_dir, runs_dir, experiment, seed)
    figures += _figure_curves(runs_dir, experiment, output_dir, seed)
    figures += _figure_ood(rows, output_dir)
    figures += _figure_compute(rows, runs_dir, experiment, output_dir, seed)
    return report, [csv_path, output_dir / "comparison.json", *figures]
