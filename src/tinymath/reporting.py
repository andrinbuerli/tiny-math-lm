"""Collect measured local runs and render comparable figures and tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
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
        decoding = summary.get("decoding", {})
        for split_name, split in summary.get("splits", {}).items():
            examples = split.get("examples")
            accuracy = split.get("accuracy")
            if examples is not None and int(examples) == 0:
                accuracy = None
            row = {
                "model": model_name,
                "arm": arm,
                "split": split_name,
                "parameters": summary.get("model_parameters"),
                "accuracy": accuracy,
                "format_validity": split.get("format_validity"),
                "mean_output_tokens": split.get("mean_output_tokens"),
                "examples": examples,
                "module_count": len(split.get("module", {})),
                "modules": ";".join(sorted(split.get("module", {}))),
                "checkpoint": summary.get("checkpoint"),
                "checkpoint_step": summary.get("checkpoint_step"),
                "decoding_strategy": decoding.get("strategy"),
                "temperature": decoding.get("temperature"),
                "max_new_tokens": decoding.get("max_new_tokens"),
                "context_length": decoding.get("context_length"),
            }
            rows.append(row)
            by_checkpoint[(model_name, arm)] = summary
    missing = {
        model: {arm: {split: True for split in SPLITS} for arm in ARMS}
        for model in MODELS
    }
    for row in rows:
        if (
            row["model"] in missing
            and row["arm"] in missing[row["model"]]
            and row["examples"]
            and int(row["examples"]) > 0
        ):
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


def _accuracy_label(rows: list[dict[str, Any]]) -> str:
    strategies = {
        row.get("decoding_strategy")
        for row in rows
        if row.get("split") == "test" and row.get("accuracy") is not None
    }
    if strategies == {"greedy"}:
        return "Greedy exact accuracy"
    if strategies == {"sampled"}:
        return "Sampled exact accuracy"
    if "greedy" in strategies and "sampled" in strategies:
        return "Exact accuracy (mixed decoding)"
    return "Exact accuracy"


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
        axis.set_ylabel(_accuracy_label(rows))
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
        axis.set_ylabel(_accuracy_label(rows))
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
                    present_positions = [
                        module_index + offset
                        for module_index, value in enumerate(values)
                        if value is not None
                    ]
                    present_values = [value for value in values if value is not None]
                    axis.bar(present_positions, present_values, width, label=f"{model} {arm}")
                index += 1
        axis.set_xticks(range(len(names)), labels, rotation=25, ha="right")
        axis.set_ylabel(_accuracy_label(rows))
        axis.set_title("Per-module ID test accuracy")
        axis.text(0.01, 0.02, "Missing measurements are omitted, not set to zero.",
                  transform=axis.transAxes, fontsize=7)
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
            positions = [pos + (index - 1) * width for pos in x]
            present_positions = [pos for pos, value in zip(positions, vals, strict=True) if value is not None]
            present_values = [value for value in vals if value is not None]
            if present_values:
                axis.bar(present_positions, present_values, width, label=split)
            for pos, value in zip(positions, vals, strict=True):
                if value is None:
                    axis.text(pos, 0.015, "missing", ha="center", va="bottom", rotation=90, fontsize=5)
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_ylabel(_accuracy_label(rows))
        axis.set_title("In-distribution and controlled OOD accuracy")
        axis.text(0.01, 0.02, "Missing measurements are labeled; no bar means no measured accuracy.",
                  transform=axis.transAxes, fontsize=7)
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
        axis.set_ylabel(_accuracy_label([row for row in rows if row.get("arm") == "sft"]))
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
        axis.set_ylabel(_accuracy_label([row for row in rows if row.get("arm") == "sft"]))
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
        result = next(
            (row for row in rows if row["model"] == path.parents[2].name and row["arm"] == arm and row["split"] == "test"),
            None,
        )
        if result is None:
            continue
        step_key = "update" if arm == "grpo" else "optimizer_step"
        checkpoint_step = result.get("checkpoint_step")
        selected = next(
            (row for row in curve if checkpoint_step is not None and row.get(step_key) == str(checkpoint_step)),
            None,
        )
        if selected is None and checkpoint_step is None:
            selected = curve[-1]
        if selected is None or token_key not in selected:
            continue
        if arm == "grpo" and selected.get("flops_estimator_version") not in {
            "dense_position_v2", "kv_cached_position_v3"
        }:
            continue
        if result and result["parameters"]:
            estimate = selected.get("estimated_flops")
            flops = (
                int(float(estimate))
                if estimate not in (None, "")
                else 6 * result["parameters"] * int(float(selected[token_key]))
            )
            points[(path.parents[2].name, arm)] = (flops, 0)
    for row in rows:
        key = (row["model"], row["arm"])
        if key in points and row["split"] == "test" and row["accuracy"] is not None:
            compute, _ = points[key]
            axis.scatter(compute, row["accuracy"], label=f"{row['model']} {row['arm']}")
    if points:
        axis.set_xscale("log")
        axis.set_xlabel("Estimated training FLOPs (dense transformer token-position model)")
        axis.set_ylabel(_accuracy_label(rows))
        axis.set_title("Accuracy versus estimated training compute")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
    else:
        _no_data(axis, "Accuracy versus compute")
    return _save(fig, output_dir, "accuracy_vs_compute")


def _format_accuracy(value: Any) -> str:
    if value is None or value == "":
        return "Missing"
    return f"{100 * float(value):.1f}%"


def _format_count(value: Any) -> str:
    if value is None or value == "":
        return "Missing"
    return f"{int(float(value)):,}"


def _data_provenance(data_bundle_dir: Path | None) -> dict[str, Any] | None:
    """Summarize the prepared dataset and actual controlled-OOD coverage."""
    if data_bundle_dir is None:
        return None
    manifest_path = data_bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_counts = manifest.get("selected_pool_category_counts", {})
    module_counts: dict[str, dict[str, int]] = {}
    for split_name in ("number_length_ood", "expression_length_ood"):
        module_counts[split_name] = {
            module: sum(int(count) for count in categories.values())
            for module, categories in selected_counts.get(split_name, {}).items()
        }
    source_counts = manifest.get("source_category_counts", {})
    return {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_revision": manifest.get("dataset_revision"),
        "split_manifest_sha256": manifest.get("split_manifest_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer", {}).get("sha256"),
        "selected_modules": manifest.get("selected_modules", []),
        "source_extrapolate_counts": {
            module: int(categories.get("extrapolate", 0))
            for module, categories in source_counts.items()
        },
        "ood_source_by_module": manifest.get("split_rules", {}).get(
            "ood_source_by_module", {}
        ),
        "ood_module_counts": module_counts,
    }


def _data_provenance_markdown(provenance: dict[str, Any] | None) -> list[str]:
    if provenance is None:
        return [
            "## Dataset and OOD coverage",
            "",
            "Prepared dataset manifest was unavailable; split provenance cannot be summarized.",
        ]
    lines = [
        "## Dataset and OOD coverage",
        "",
        f"- Dataset: `{provenance.get('dataset_id')}` at revision `{provenance.get('dataset_revision')}`.",
        f"- Split manifest SHA-256: `{provenance.get('split_manifest_sha256')}`.",
        f"- Tokenizer SHA-256: `{provenance.get('tokenizer_sha256')}`.",
        f"- Selected modules: {', '.join(provenance.get('selected_modules', []))}.",
        "- OOD examples by module:",
        "",
        "| OOD split | Module example counts |",
        "|---|---|",
    ]
    for split in ("number_length_ood", "expression_length_ood"):
        counts = provenance["ood_module_counts"].get(split, {})
        listed = "; ".join(f"{module}: {count:,}" for module, count in sorted(counts.items()))
        lines.append(f"| {split} | {listed or 'No eligible examples'} |")

    extrapolate = provenance.get("source_extrapolate_counts", {})
    sources = provenance.get("ood_source_by_module", {})
    if extrapolate and not any(extrapolate.values()):
        lines.extend(
            [
                "",
                (
                    "No official `extrapolate` examples were present in the selected source modules. "
                    "Controlled OOD examples were reserved from training categories and kept out of "
                    "the SFT and RLVR pools."
                ),
            ]
        )
    elif any("controlled fallback" in source for source in sources.values()):
        fallback_modules = [
            module for module, source in sources.items() if "controlled fallback" in source
        ]
        lines.extend(
            [
                "",
                "Controlled OOD examples for these modules were reserved from training categories: "
                + ", ".join(fallback_modules)
                + ".",
            ]
        )
    return lines


def _read_training_metrics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _training_metric_path(
    runs_dir: Path, experiment: str, model: str, arm: str, seed: int | None
) -> Path | None:
    arm_dir = runs_dir / experiment / model / arm
    if seed is not None:
        path = arm_dir / f"seed{seed}" / "metrics.csv"
        return path if path.is_file() else None
    matches = sorted(arm_dir.glob("seed*/metrics.csv"))
    return matches[-1] if matches else None


def _accuracy_gain(
    sft: dict[str, Any] | None, candidate: dict[str, Any] | None
) -> str:
    if not sft or not candidate or sft.get("accuracy") is None or candidate.get("accuracy") is None:
        return "Missing"
    return f"{100 * (float(candidate['accuracy']) - float(sft['accuracy'])):+.1f} pp"


def _average_percent(metrics: list[dict[str, str]], column: str) -> str:
    values = [float(row[column]) for row in metrics if row.get(column) not in (None, "")]
    return f"{100 * sum(values) / len(values):.1f}%" if values else "Missing"


def _format_ood_measurement(row: dict[str, Any] | None) -> str:
    if not row or not row.get("examples"):
        return "Missing / 0 examples / no modules"
    modules = row.get("modules") or "module details unavailable"
    return (
        f"{_format_accuracy(row.get('accuracy'))} / {_format_count(row.get('examples'))} examples / "
        f"{modules}"
    )


def _markdown_report(
    *,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
    runs_dir: Path,
    experiment: str,
    seed: int | None,
) -> str:
    measured = {
        (row["model"], row["arm"], row["split"]): row
        for row in rows
    }
    actual_combinations = sum(
        not report["coverage"]["missing"][model][arm][split]
        for model in MODELS
        for arm in ARMS
        for split in SPLITS
    )
    decoding_protocols = sorted(
        {
            (
                row.get("decoding_strategy"),
                row.get("temperature"),
                row.get("max_new_tokens"),
                row.get("context_length"),
            )
            for row in rows
        },
        key=lambda protocol: tuple("" if value is None else str(value) for value in protocol),
    )
    if decoding_protocols:
        decoding_lines = [
            (
                f"- {strategy} decoding at temperature {temperature}; max_new_tokens cap {cap}; "
                f"context length {context_length}. The effective cap is also limited by each "
                "prompt's remaining context."
                if strategy is not None
                else "- Decoding metadata is missing for at least one saved evaluation."
            )
            for strategy, temperature, cap, context_length in decoding_protocols
        ]
    else:
        decoding_lines = ["Decoding metadata is missing because no evaluation results are available."]
    lines = [
        f"# TinyMath-RL comparison: `{experiment}`",
        "",
        f"Seed: `{seed}`" if seed is not None else "Seed filter: all available seeds",
        f"Evaluation coverage: {actual_combinations}/{len(MODELS) * len(ARMS) * len(SPLITS)} model/arm/split measurements.",
        "Only saved evaluation results are included; `Missing` means no result file was available.",
        "",
        *_data_provenance_markdown(report.get("data_provenance")),
        "",
        "## Decoding protocol",
        "",
        *decoding_lines,
        "",
        "## In-distribution test accuracy",
        "",
        "| Model | Parameters | SFT | GRPO | Continued SFT | GRPO gain vs SFT | Continued SFT gain vs SFT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        sft = measured.get((model, "sft", "test"))
        grpo = measured.get((model, "grpo", "test"))
        continued = measured.get((model, "continued_sft", "test"))
        parameter_count = next(
            (row.get("parameters") for row in rows if row["model"] == model and row.get("parameters")),
            None,
        )

        lines.append(
            "| {model} | {parameters} | {sft} | {grpo} | {continued} | {grpo_gain} | {continued_gain} |".format(
                model=model,
                parameters=_format_count(parameter_count),
                sft=_format_accuracy(sft.get("accuracy") if sft else None),
                grpo=_format_accuracy(grpo.get("accuracy") if grpo else None),
                continued=_format_accuracy(continued.get("accuracy") if continued else None),
                grpo_gain=_accuracy_gain(sft, grpo),
                continued_gain=_accuracy_gain(sft, continued),
            )
        )

    lines.extend(
        [
            "",
            "## Controlled OOD accuracy",
            "",
            "| Model | Arm | Number length: accuracy / examples / modules | Expression length: accuracy / examples / modules |",
            "|---|---|---|---|",
        ]
    )
    for model in MODELS:
        for arm in ARMS:
            number = measured.get((model, arm, "number_length_ood"))
            expression = measured.get((model, arm, "expression_length_ood"))
            lines.append(
                f"| {model} | {arm} | {_format_ood_measurement(number)} | "
                f"{_format_ood_measurement(expression)} |"
            )

    lines.extend(
        [
            "",
            "## GRPO rollout diagnostics",
            "",
            "Mixed-group rate is averaged over recorded updates; these are groups with both correct and incorrect outcomes.",
            "",
            "| Model | Updates | Mean reward | Mixed-group rate | All-correct rate | All-wrong rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        metrics_path = _training_metric_path(runs_dir, experiment, model, "grpo", seed)
        metrics = _read_training_metrics(metrics_path) if metrics_path else []

        latest_update = metrics[-1].get("update") if metrics else None
        lines.append(
            f"| {model} | {_format_count(latest_update)} | {_average_percent(metrics, 'reward/mean')} | "
            f"{_average_percent(metrics, 'group/mixed_rate')} | "
            f"{_average_percent(metrics, 'group/all_correct_rate')} | "
            f"{_average_percent(metrics, 'group/all_wrong_rate')} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy efficiency",
            "",
            "Efficiency pairs ID test accuracy with each arm's logged token count and estimated FLOPs.",
            "FLOPs are a dense-transformer approximation, not hardware-counter measurements.",
            "",
            "| Model | Arm | ID accuracy | Logged tokens | Estimated FLOPs | ID percentage points / 1M logged tokens | ID percentage points / 1T FLOPs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        for arm in ARMS:
            evaluation = measured.get((model, arm, "test"))
            metrics_path = _training_metric_path(runs_dir, experiment, model, arm, seed)
            metrics = _read_training_metrics(metrics_path) if metrics_path else []
            checkpoint_step = evaluation.get("checkpoint_step") if evaluation else None
            step_key = "update" if arm == "grpo" else "optimizer_step"
            selected = next(
                (row for row in metrics if checkpoint_step is not None and row.get(step_key) == str(checkpoint_step)),
                None,
            )
            if selected is None and checkpoint_step is None and metrics:
                selected = metrics[-1]
            latest = selected or {}
            accuracy = evaluation.get("accuracy") if evaluation else None
            token_positions = latest.get("tokens_seen")
            estimated_flops = latest.get("estimated_flops")
            token_efficiency = flops_efficiency = "Missing"
            if accuracy is not None and token_positions not in (None, ""):
                count = float(token_positions)
                if count > 0:
                    token_efficiency = f"{100 * float(accuracy) / (count / 1_000_000):.3g}"
            if accuracy is not None and estimated_flops not in (None, ""):
                flops = float(estimated_flops)
                if math.isfinite(flops) and flops > 0:
                    flops_efficiency = f"{100 * float(accuracy) / (flops / 1_000_000_000_000):.3g}"
            lines.append(
                f"| {model} | {arm} | {_format_accuracy(accuracy)} | {_format_count(token_positions)} | "
                f"{_format_count(estimated_flops)} | {token_efficiency} | {flops_efficiency} |"
            )

    lines.extend(["", "## Limitations", ""])
    if seed is not None:
        lines.append(f"- This report covers seed `{seed}` only; it does not estimate between-seed uncertainty.")
    lines.extend(
        [
            "- OOD results describe the fixed prepared number-length and expression-length pools; unavailable pools remain missing.",
            "- Training-time train/validation/test greedy-accuracy curves come from the same fixed qualitative probe: 2 examples per module across 5 modules (10 examples per split). Read these as a small-sample inspection signal, not full-split accuracy; the example identities stay fixed across checkpoints.",
            "- SFT token counts include processed input positions; GRPO token counts include generated completion tokens, so token efficiency is descriptive within each arm, not a strictly matched cross-arm measure.",
            "- Continued SFT keeps the configured optimizer-step count and uses four-way gradient accumulation to approximate the GRPO arm's additional compute; compare logged FLOP estimates because rollout lengths vary and the effective batch differs.",
            "- FLOP and token efficiency are descriptive proxies tied to the logged counters; they are not hardware energy or latency measurements.",
            "- Accuracy differences are descriptive for this protocol and seed; they do not establish a general scaling or RLVR effect beyond these runs.",
            "",
        ]
    )
    return "\n".join(lines)


def make_report(
    *,
    runs_dir: Path,
    experiment: str,
    output_dir: Path,
    seed: int | None = None,
    data_bundle_dir: Path | None = None,
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
                        "examples": "", "module_count": 0, "modules": "",
                        "checkpoint": "", "checkpoint_step": "",
                        "decoding_strategy": "", "temperature": "", "max_new_tokens": "",
                        "context_length": "",
                    })
                )
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "arm", "split", "parameters", "accuracy", "format_validity",
                  "mean_output_tokens", "examples", "module_count", "modules", "checkpoint",
                  "checkpoint_step", "decoding_strategy", "temperature", "max_new_tokens",
                  "context_length"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(expected_rows)
    report = {
        "experiment": experiment,
        "seed": seed,
        "available_measurements": rows,
        "coverage": coverage,
        "data_provenance": _data_provenance(data_bundle_dir),
        "compute_estimate": "SFT uses 6PN dense token-position approximation; GRPO records per-update token-position estimates for the rollout decoder and policy/reference passes",
        "limitations": [
            "Compute uses a dense-transformer token-position approximation; it is not hardware-counter FLOPs.",
            "GRPO compute estimates identify the decoding estimator used; a resumed run can contain both pre-cache and KV-cached segments.",
            "Training-time train/validation/test greedy-accuracy curves use the same fixed qualitative probe of 2 examples per module across 5 modules (10 examples per split), not full-split accuracy; selected example identities remain fixed across checkpoints.",
        ],
    }
    report_path = output_dir / "report.md"
    report_path.write_text(
        _markdown_report(
            report=report,
            rows=rows,
            runs_dir=runs_dir,
            experiment=experiment,
            seed=seed,
        ),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
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
    return report, [csv_path, output_dir / "comparison.json", report_path, *figures]


# Experiment 11 has a different question from the historical three-arm report:
# its rows compare a frozen answer-only checkpoint with a reasoning-trained
# checkpoint, and incomplete evaluation is a first-class result. Keep the
# renderer separate so the established report and plots remain stable.
EXP11_SPLITS = ("test", "number_length_ood", "expression_length_ood")
EXP11_STATUS = ("missing", "not-run", "running", "partial", "failed", "complete")
EXP11_SELECTORS = {"sft": "best", "grpo": "latest", "continued_sft": "best"}


def _json_file(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wandb_provenance(run_dir: Path) -> dict[str, Any]:
    """Expose local W&B identity and sync state without implying an online link."""
    state = _json_file(run_dir / "wandb_run.json") or {}
    identity = state.get("identity") or {}
    url = state.get("url")
    if state.get("mode") == "offline" and state.get("sync_status") != "synced":
        url = None
    return {
        "run_id": state.get("id"),
        "run_name": identity.get("name"),
        "project": state.get("project") or identity.get("project"),
        "group": identity.get("group"),
        "phase": (identity.get("metadata") or {}).get("phase"),
        "mode": state.get("mode"),
        "sync_status": state.get("sync_status"),
        "url": url,
        "local_path": str(run_dir),
    }


def _question_identity(path: Path) -> tuple[str | None, int]:
    """Hash the ordered module/question/target identity in a prediction CSV."""
    if not path.is_file():
        return None, 0
    digest = hashlib.sha256()
    count = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = [row.get("module", ""), row.get("question", ""), row.get("target", "")]
            digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
            count += 1
    return (digest.hexdigest(), count) if count else (None, 0)


def _exp11_result_path(
    runs_dir: Path, experiment: str, model: str, arm: str, seed: int,
    selector: str | None = None, expected_checkpoint: Path | None = None,
) -> Path | None:
    root = runs_dir / experiment / model / "evaluation" / arm
    if selector is not None:
        search_root = root / selector
        matches = [search_root / f"seed{seed}" / "results.json"]
        matches.extend(sorted(search_root.glob(f"*/seed{seed}/results.json")))
    else:
        matches = sorted(root.glob(f"*/*/seed{seed}/results.json"))
    for result_path in reversed(matches):
        if not result_path.is_file():
            continue
        if expected_checkpoint is None or _evaluation_matches_checkpoint(
            result_path.parent, expected_checkpoint
        ):
            return result_path
    return None


def _evaluation_matches_checkpoint(evaluation_dir: Path, checkpoint: Path) -> bool:
    """Keep phase-specific pilot evaluations out of the main experiment rows."""
    expected = checkpoint.resolve()
    for filename in ("results.json", "resolved_config.json"):
        value = _json_file(evaluation_dir / filename)
        saved = value.get("checkpoint") if value else None
        if saved:
            try:
                return Path(str(saved)).resolve() == expected
            except (OSError, RuntimeError):
                return False
    return False


def _exp11_eval_dir(
    runs_dir: Path, experiment: str, model: str, arm: str, seed: int,
    selector: str | None, expected_checkpoint: Path | None = None,
) -> Path:
    root = runs_dir / experiment / model / "evaluation" / arm
    candidates: list[Path] = []
    if selector:
        selected = root / selector / f"seed{seed}"
        candidates.append(selected)
        candidates.extend(sorted((root / selector).glob(f"*/seed{seed}")))
    else:
        candidates.extend(sorted(root.glob(f"*/*/seed{seed}")))
    if expected_checkpoint is not None:
        candidates = [
            path for path in candidates
            if path.is_dir() and _evaluation_matches_checkpoint(path, expected_checkpoint)
        ]
    else:
        candidates = [path for path in candidates if path.exists()]
    if candidates:
        return candidates[-1]
    return root / str(selector or "best") / f"seed{seed}"


def _exp11_selector(
    snapshot: dict[str, Any], model: str, arm: str, baseline_name: str
) -> str | None:
    provenance = snapshot.get("provenance", {}).get("baseline_evaluation_wandb_runs", {})
    saved = provenance.get(model, {}).get(arm, {}).get("checkpoint_selector")
    if saved:
        return str(saved)
    choices = sorted((Path(baseline_name) / model / "evaluation" / arm).glob("*/seed18/results.json"))
    return choices[-1].parent.parent.name if choices else None


def _exp11_state(
    result_path: Path | None, evaluation_root: Path, split: str, expected_examples: int | None
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    result = _json_file(result_path) if result_path else None
    progress_path = evaluation_root / "progress.json"
    progress = _json_file(progress_path)
    state = _json_file(evaluation_root / "wandb_run.json")
    saved_split = (result or {}).get("splits", {}).get(split)
    if saved_split is not None:
        count = int(saved_split.get("examples") or 0)
        declared_expected = (saved_split.get("expected_examples")
                             or saved_split.get("total_examples")
                             or (result or {}).get("expected_examples"))
        required = expected_examples if expected_examples is not None else declared_expected
        if required is not None and count >= int(required):
            return "complete", result, saved_split
        if (state or {}).get("run_status") == "failed":
            return "failed", result, saved_split
        return "partial", result, saved_split
    scored = (progress or {}).get("scored", {}).get(split, [])
    if scored:
        return "partial", result, {"examples": len(scored)}
    status = (state or {}).get("run_status")
    if status == "failed" or (state or {}).get("exit_code", 0) not in (None, 0):
        return "failed", result, None
    if status == "running":
        return "running", result, None
    if result is not None:
        return "partial", result, None
    return "not-run", None, None


def _metric_at_checkpoint(metrics_path: Path, arm: str, checkpoint_step: Any) -> dict[str, str]:
    metrics = _read_training_metrics(metrics_path)
    step_key = "update" if arm == "grpo" else "optimizer_step"
    if checkpoint_step is not None:
        matching = next((row for row in metrics if row.get(step_key) == str(checkpoint_step)), None)
        return matching or {}
    return metrics[-1] if metrics else {}


def _exp11_accuracy(value: Any) -> str:
    return "Missing" if value is None else f"{100 * float(value):.2f}%"


def _mean_training_metric(metrics: list[dict[str, str]], column: str) -> tuple[float | None, int]:
    values: list[float] = []
    for row in metrics:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def _efficiency_score(accuracy: Any, counter: Any, scale: float) -> float | None:
    """Return exact-accuracy percentage points per scaled token/FLOP counter."""
    if accuracy in (None, "") or counter in (None, ""):
        return None
    try:
        acc_value = float(accuracy)
        counter_value = float(counter)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(acc_value) or not math.isfinite(counter_value) or counter_value <= 0:
        return None
    return 100.0 * acc_value / (counter_value / scale)


def _baseline_wandb_provenance(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Copy the baseline evaluation identities needed to audit its snapshot."""
    saved = snapshot.get("provenance", {}).get("baseline_evaluation_wandb_runs", {})
    output: dict[str, Any] = {}
    for model, arms in saved.items():
        output[model] = {}
        for arm, run in arms.items():
            output[model][arm] = {
                key: run.get(key)
                for key in (
                    "id", "path", "mode", "sync_status", "run_status", "exit_code",
                    "checkpoint_selector", "checkpoint_step", "splits",
                )
                if run.get(key) is not None
            }
    return output


def _exp11_rlvr_gains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare GRPO with SFT only when both split measurements share identities."""
    by_key = {(row["model"], row["arm"], row["split"]): row for row in rows}
    gains: list[dict[str, Any]] = []
    for model in MODELS:
        for split in EXP11_SPLITS:
            sft = by_key.get((model, "sft", split), {})
            grpo = by_key.get((model, "grpo", split), {})
            status = "pending"
            delta = None
            reason = "Both SFT and GRPO evaluations must be complete and use the same questions."
            if sft.get("exp11_status") == "failed" or grpo.get("exp11_status") == "failed":
                status = "failed"
                reason = "At least one required evaluation failed."
            elif sft.get("exp11_status") == "complete" and grpo.get("exp11_status") == "complete":
                sft_identity = sft.get("exp11_question_identity_sha256")
                grpo_identity = grpo.get("exp11_question_identity_sha256")
                if sft_identity and grpo_identity and sft_identity == grpo_identity:
                    sft_accuracy = sft.get("exp11_accuracy")
                    grpo_accuracy = grpo.get("exp11_accuracy")
                    if sft_accuracy is not None and grpo_accuracy is not None:
                        status = "measured"
                        delta = float(grpo_accuracy) - float(sft_accuracy)
                        reason = "Complete results share the same ordered module/question/target identities."
                    else:
                        reason = "Complete evaluations lack accuracy values."
                else:
                    status = "identity-unverified"
                    reason = "Complete evaluations do not have matching ordered question identities."
            gains.append({
                "model": model,
                "split": split,
                "status": status,
                "grpo_minus_sft_accuracy_delta": delta,
                "grpo_minus_sft_percentage_points": 100.0 * delta if delta is not None else None,
                "reason": reason,
            })
    return gains


def _exp11_continued_sft_gains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare continued SFT with SFT only for complete, identity-matched splits."""
    by_key = {(row["model"], row["arm"], row["split"]): row for row in rows}
    gains: list[dict[str, Any]] = []
    for model in MODELS:
        for split in EXP11_SPLITS:
            sft = by_key.get((model, "sft", split), {})
            control = by_key.get((model, "continued_sft", split), {})
            status = "pending"
            delta = None
            reason = "Both SFT and continued-SFT evaluations must be complete and use the same questions."
            if sft.get("exp11_status") == "failed" or control.get("exp11_status") == "failed":
                status, reason = "failed", "At least one required evaluation failed."
            elif sft.get("exp11_status") == "complete" and control.get("exp11_status") == "complete":
                left, right = sft.get("exp11_question_identity_sha256"), control.get("exp11_question_identity_sha256")
                if left and right and left == right:
                    if sft.get("exp11_accuracy") is not None and control.get("exp11_accuracy") is not None:
                        status = "measured"
                        delta = float(control["exp11_accuracy"]) - float(sft["exp11_accuracy"])
                        reason = "Complete results share the same ordered module/question/target identities."
                    else:
                        reason = "Complete evaluations lack accuracy values."
                else:
                    status, reason = "identity-unverified", "Complete evaluations do not have matching ordered question identities."
            gains.append({
                "model": model, "split": split, "status": status,
                "continued_sft_minus_sft_accuracy_delta": delta,
                "continued_sft_minus_sft_percentage_points": 100 * delta if delta is not None else None,
                "reason": reason,
            })
    return gains


def _exp11_charts(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for split, filename, title in (
        ("test", "id_accuracy", "Experiment 11: ID test accuracy"),
        ("ood", "ood_accuracy", "Experiment 11: controlled OOD accuracy"),
    ):
        selected_splits = ("test",) if split == "test" else ("number_length_ood", "expression_length_ood")
        fig, axis = plt.subplots(figsize=(9, 5))
        selected = [row for row in rows if row["split"] in selected_splits]
        positions = list(range(len(MODELS) * len(ARMS)))
        labels = [f"{model}\n{arm}" for model in MODELS for arm in ARMS]
        groups = [("Experiment 10 baseline", "baseline_accuracy", "test", "#777777")]
        if split == "test":
            groups.append(("Experiment 11 reasoning", "exp11_accuracy", "test", "#2878b5"))
        else:
            groups.extend([
                ("Experiment 10 number-length OOD", "baseline_accuracy", "number_length_ood", "#777777"),
                ("Experiment 10 expression-length OOD", "baseline_accuracy", "expression_length_ood", "#999999"),
                ("Experiment 11 number-length OOD", "exp11_accuracy", "number_length_ood", "#2878b5"),
                ("Experiment 11 expression-length OOD", "exp11_accuracy", "expression_length_ood", "#72a6cf"),
            ])
        bar_count = len(groups)
        width = 0.8 / bar_count
        for index, (label, field, group_split, color) in enumerate(groups):
            vals = []
            for model in MODELS:
                for arm in ARMS:
                    row = next((r for r in rows if r["model"] == model and r["arm"] == arm
                                and r["split"] == group_split), None)
                    vals.append(row.get(field) if row else None)
            offset = (index - (bar_count - 1) / 2) * width
            present = [(pos + offset, value) for pos, value in zip(positions, vals, strict=True) if value is not None]
            if present:
                axis.bar([item[0] for item in present], [item[1] for item in present], width,
                         color=color, label=label)
            for pos, value in zip(positions, vals, strict=True):
                if value is None:
                    axis.text(pos + offset, 0.01, "missing", fontsize=5, rotation=90, ha="center")
        if any(value is not None for row in selected for value in (row.get("baseline_accuracy"), row.get("exp11_accuracy"))):
            axis.set_xticks(positions, labels, fontsize=7)
            axis.set_ylabel("Greedy exact accuracy")
            axis.set_title(title)
            axis.set_ylim(0, 1.05)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=8)
        else:
            _no_data(axis, title + " (no measurements yet)")
        outputs.extend(_save(fig, output_dir, filename))
    return outputs


def _exp11_breakdown_charts(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Render breakdowns from the already checkpoint-filtered report rows."""
    outputs: list[Path] = []
    specs = (
        ("module_metrics", "per_module_accuracy", "Per-module ID test accuracy", "module"),
        ("by_digit_width", "accuracy_by_digit_width", "Accuracy by operand digit width", "Longest decimal operand width"),
        ("by_operator_count", "accuracy_by_expression_length", "Accuracy by expression length", "Operator count"),
    )
    for field, filename, title, x_label in specs:
        fig, axis = plt.subplots(figsize=(9, 5))
        found = False
        selected = [row for row in rows if row["split"] == "test"]
        if field == "module_metrics":
            categories = sorted({category for row in selected for category in row.get(field, {})})
            width = 0.8 / max(1, len(MODELS) * len(ARMS))
            for idx, (model, arm) in enumerate((m, a) for m in MODELS for a in ARMS):
                row = next((r for r in selected if r["model"] == model and r["arm"] == arm), {})
                vals = [row.get(field, {}).get(category, {}).get("accuracy") for category in categories]
                if any(value is not None for value in vals):
                    found = True
                    offset = (idx - (len(MODELS) * len(ARMS) - 1) / 2) * width
                    axis.bar([i + offset for i, value in enumerate(vals) if value is not None],
                             [value for value in vals if value is not None], width, label=f"{model} {arm}")
            if found:
                axis.set_xticks(range(len(categories)), [c.split("__")[-1] for c in categories], rotation=25, ha="right")
        else:
            for row in selected:
                curve = row.get(field, {}) or {}
                if not curve:
                    continue
                values = sorted(((float(key), metric.get("accuracy")) for key, metric in curve.items()), key=lambda item: item[0])
                values = [(x, y) for x, y in values if y is not None]
                if values:
                    found = True
                    axis.plot([x for x, _ in values], [y for _, y in values], marker="o",
                              label=f"{row['model']} {row['arm']}")
        if found:
            axis.set_xlabel(x_label)
            axis.set_ylabel("Exact accuracy")
            axis.set_title(title)
            axis.legend(fontsize=7, ncol=2)
            axis.grid(alpha=0.25)
        else:
            _no_data(axis, title)
        outputs.extend(_save(fig, output_dir, filename))
    return outputs


def _exp11_grpo_sample_efficiency_chart(
    *, runs_dir: Path, experiment: str, output_dir: Path, seed: int
) -> list[Path]:
    """Plot per-update GRPO reward and informative-group rate against generated tokens."""
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    colors = plt.get_cmap("tab10")
    for index, model in enumerate(MODELS):
        metrics_path = runs_dir / experiment / model / "grpo" / f"seed{seed}" / "metrics.csv"
        reward_points: list[tuple[float, float]] = []
        informative_points: list[tuple[float, float]] = []
        for row in _read_training_metrics(metrics_path):
            try:
                tokens = float(row["tokens_seen"])
                reward = float(row["reward/mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(tokens) or tokens < 0 or not math.isfinite(reward):
                continue
            reward_points.append((tokens, reward))
            mixed_value = row.get("group/mixed_rate")
            if mixed_value not in (None, ""):
                try:
                    informative = float(mixed_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(informative):
                    informative_points.append((tokens, informative))
        if not reward_points:
            continue
        plotted = True
        color = colors(index)
        axis.plot(
            [point[0] for point in reward_points],
            [point[1] for point in reward_points],
            color=color,
            label=f"{model} reward/mean",
        )
        if informative_points:
            axis.plot(
                [point[0] for point in informative_points],
                [point[1] for point in informative_points],
                color=color,
                linestyle="--",
                alpha=0.8,
                label=f"{model} informative-group rate",
            )
    if plotted:
        axis.set_xlabel("Cumulative generated tokens (tokens_seen; symlog scale)")
        axis.set_xscale("symlog", linthresh=1_000)
        axis.set_ylabel("Rate")
        axis.set_ylim(0, 1.05)
        axis.set_title("GRPO reward and informative groups versus generated tokens")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    else:
        _no_data(axis, "GRPO sample efficiency (no per-update reward/token metrics)")
    return _save(fig, output_dir, "grpo_sample_efficiency")


def make_exp11_report(
    *,
    runs_dir: Path,
    output_dir: Path,
    seed: int = 18,
    experiment: str = "exp11_reasoning_500k_100k",
    baseline_snapshot: Path | None = None,
    baseline_runs_dir: Path | None = None,
    annotation_bundle_dir: Path | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """Create the continuously refreshable Experiment 11 status and comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_snapshot = baseline_snapshot or Path("baselines/exp10_restarted_100k_greedy_restart_2/seed18/snapshot.json")
    snapshot = _json_file(baseline_snapshot) or {}
    try:
        baseline_snapshot_sha256 = hashlib.sha256(baseline_snapshot.read_bytes()).hexdigest()
    except OSError:
        baseline_snapshot_sha256 = None
    baseline_name = str(snapshot.get("experiment", {}).get("canonical_name", "restarted_100k_greedy_restart_2"))
    baseline_wandb_runs = _baseline_wandb_provenance(snapshot)
    baseline_runs_dir = baseline_runs_dir or runs_dir
    baseline_records = {
        (item.get("model"), item.get("arm"), item.get("split")): item
        for item in snapshot.get("available_measurements", [])
    }
    rows: list[dict[str, Any]] = []
    raw_results: dict[tuple[str, str], tuple[Path | None, dict[str, Any] | None]] = {}
    training_metrics_by_arm: dict[tuple[str, str], dict[str, str]] = {}
    for model in MODELS:
        for arm in ARMS:
            training_run_dir = runs_dir / experiment / model / arm / f"seed{seed}"
            baseline_selector = _exp11_selector(snapshot, model, arm, baseline_name)
            base_eval = _exp11_result_path(
                baseline_runs_dir, baseline_name, model, arm, seed, baseline_selector
            )
            selector = EXP11_SELECTORS[arm]
            main_checkpoint_name = "latest.pt" if arm == "grpo" else "best.pt"
            main_checkpoint = (
                runs_dir / experiment / model / arm / f"seed{seed}" / main_checkpoint_name
            )
            # Each main row is tied to the canonical full-run checkpoint path.
            # This excludes smoke and pilot checkpoints even before the main
            # checkpoint has been created.
            expected_main_checkpoint = main_checkpoint
            exp_result_path = _exp11_result_path(
                runs_dir, experiment, model, arm, seed, selector,
                expected_checkpoint=expected_main_checkpoint,
            ) or _exp11_result_path(
                runs_dir, experiment, model, arm, seed,
                expected_checkpoint=expected_main_checkpoint,
            )
            if exp_result_path is not None:
                result = _json_file(exp_result_path)
                selector = exp_result_path.parents[1].name
            else:
                result = None
            raw_results[(model, arm)] = (exp_result_path, result)
            train_metrics = _metric_at_checkpoint(
                training_run_dir / "metrics.csv",
                arm,
                (result or {}).get("checkpoint_step"),
            )
            training_metrics_by_arm[(model, arm)] = train_metrics
            baseline_record_for_training = next(
                (baseline_records.get((model, arm, split_name)) for split_name in EXP11_SPLITS
                 if baseline_records.get((model, arm, split_name)) is not None),
                None,
            )
            baseline_train_metrics = _metric_at_checkpoint(
                baseline_runs_dir / baseline_name / model / arm / f"seed{seed}" / "metrics.csv",
                arm,
                (baseline_record_for_training or {}).get("checkpoint_step"),
            )
            def estimator_summary(path: Path) -> str | None:
                versions = sorted({
                    row.get("flops_estimator_version") for row in _read_training_metrics(path)
                    if row.get("flops_estimator_version")
                })
                if not versions:
                    return None
                return versions[0] if len(versions) == 1 else "mixed:" + "+".join(versions)

            current_flops_estimator = estimator_summary(training_run_dir / "metrics.csv")
            baseline_flops_estimator = estimator_summary(
                baseline_runs_dir / baseline_name / model / arm / f"seed{seed}" / "metrics.csv"
            )
            same_flops_estimator = (
                current_flops_estimator == baseline_flops_estimator
                and (current_flops_estimator is not None or arm != "grpo")
            )
            def numeric_delta(current: Any, prior: Any) -> float | None:
                if current in (None, "") or prior in (None, ""):
                    return None
                try:
                    return float(current) - float(prior)
                except (TypeError, ValueError):
                    return None

            step_key = "update" if arm == "grpo" else "optimizer_step"
            update_matched = (
                train_metrics.get(step_key) not in (None, "")
                and train_metrics.get(step_key) == baseline_train_metrics.get(step_key)
            )
            additional_tokens = (
                numeric_delta(train_metrics.get("tokens_seen"), baseline_train_metrics.get("tokens_seen"))
                if update_matched else None
            )
            additional_flops = (
                numeric_delta(train_metrics.get("estimated_flops"), baseline_train_metrics.get("estimated_flops"))
                if update_matched and same_flops_estimator else None
            )
            for split in EXP11_SPLITS:
                key = (model, arm, split)
                baseline = baseline_records.get(key)
                # A snapshot row is the evidence for a baseline measurement. Its
                # coverage map can explain the omission, but cannot supply a value.
                baseline_count = int((baseline or {}).get("examples") or 0)
                baseline_status = (
                    "complete" if baseline_count > 0 else "partial" if baseline is not None else "missing"
                )
                base_dir = (base_eval.parent if base_eval else
                            baseline_runs_dir / baseline_name / model / "evaluation" / arm / str(baseline_selector or "") / f"seed{seed}")
                base_id = _question_identity(base_dir / f"{split}_predictions.csv")
                eval_root = exp_result_path.parent if exp_result_path else _exp11_eval_dir(
                    runs_dir, experiment, model, arm, seed, selector,
                    expected_checkpoint=expected_main_checkpoint,
                )
                expected = int(baseline.get("examples") or 0) if baseline else None
                status, result, split_data = _exp11_state(exp_result_path, eval_root, split, expected)
                if result is None:
                    result = raw_results[(model, arm)][1]
                exp_id = _question_identity(eval_root / f"{split}_predictions.csv")
                identity_matches = bool(base_id[0] and exp_id[0] and base_id == exp_id)
                accuracy = split_data.get("accuracy") if split_data else None
                if status == "complete" and split_data and int(split_data.get("examples") or 0) == 0:
                    status = "partial"
                delta = None
                if status == "complete" and baseline and accuracy is not None and baseline.get("accuracy") is not None and identity_matches:
                    delta = float(accuracy) - float(baseline["accuracy"])
                row = {
                    "model": model, "arm": arm, "split": split,
                    "parameters": (result or {}).get("model_parameters"),
                    "baseline_status": baseline_status,
                    "baseline_accuracy": baseline.get("accuracy") if baseline else None,
                    "baseline_examples": baseline.get("examples") if baseline else None,
                    "exp11_status": status,
                    "exp11_accuracy": accuracy,
                    "delta_accuracy": delta,
                    "delta_status": "comparable" if delta is not None else "identity-unverified-or-incomplete",
                    "same_question_identity": identity_matches,
                    "baseline_question_identity_sha256": base_id[0],
                    "exp11_question_identity_sha256": exp_id[0],
                    "baseline_question_count": base_id[1], "exp11_question_count": exp_id[1],
                    "examples": split_data.get("examples") if split_data else None,
                    "format_validity": split_data.get("format_validity") if split_data else None,
                    "trace_structure_validity": split_data.get("trace_structure_validity") if split_data else None,
                    "trace_reference_exact_match_rate": (
                        split_data.get("trace_reference_exact_match_rate") if split_data else None
                    ),
                    "module_metrics": split_data.get("module", {}) if split_data else {},
                    "by_digit_width": split_data.get("by_digit_width", {}) if split_data else {},
                    "by_operator_count": split_data.get("by_operator_count", {}) if split_data else {},
                    "capped_completions": split_data.get("capped_completions") if split_data else None,
                    "mean_output_tokens": split_data.get("mean_output_tokens") if split_data else None,
                    "checkpoint_step": (result or {}).get("checkpoint_step"),
                    "selector": selector,
                    "logged_tokens": train_metrics.get("tokens_seen"),
                    "estimated_flops": train_metrics.get("estimated_flops"),
                    "flops_estimator": current_flops_estimator,
                    "baseline_logged_tokens": baseline_train_metrics.get("tokens_seen"),
                    "baseline_estimated_flops": baseline_train_metrics.get("estimated_flops"),
                    "baseline_flops_estimator": baseline_flops_estimator,
                    "same_flops_estimator": same_flops_estimator,
                    "additional_tokens": additional_tokens,
                    "additional_flops": additional_flops,
                    "training_wandb": _wandb_provenance(training_run_dir),
                    "evaluation_wandb": _wandb_provenance(eval_root),
                }
                rows.append(row)

    annotation_bundle_dir = annotation_bundle_dir or Path(
        "data/processed/exp11_reasoning_500k_100k-seed18"
    )
    annotation_manifest = _json_file(annotation_bundle_dir / "manifest.json") or {}
    diagnostics = {
        "protocol": annotation_manifest.get("tokenizer", {}).get("protocol", "reasoning_v1"),
        "annotation_sha256": annotation_manifest.get("annotation", {}).get("annotation_sha256"),
        "annotation_provenance": {
            "source_bundle": annotation_manifest.get("annotation", {}).get("source_bundle"),
            "source_manifest_sha256": annotation_manifest.get("annotation", {}).get("source_manifest_sha256"),
            "source_split_sha256": annotation_manifest.get("annotation", {}).get("source_split_sha256"),
            "source_tokenizer_sha256": annotation_manifest.get("annotation", {}).get("source_tokenizer_sha256"),
            "tokenizer_sha256": annotation_manifest.get("annotation", {}).get("tokenizer_sha256")
                or annotation_manifest.get("tokenizer", {}).get("sha256"),
        },
        "tokenizer": annotation_manifest.get("tokenizer", {}),
        "module_split_statistics": annotation_manifest.get("annotation", {}).get("module_split_statistics", {}),
        "collapsed_categories": annotation_manifest.get("annotation", {}).get("collapsed_categories", {}),
        "full_length_statistics": {},
        "length_metrics": {},
    }
    # The annotation manifest stores per-module/split reduction counts. Count the
    # individual collapse categories from rows when the annotated JSONL exists.
    collapsed_counts: dict[str, int] = {}
    if annotation_bundle_dir.is_dir():
        for annotated_path in sorted(annotation_bundle_dir.glob("*.jsonl")):
            with annotated_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    module = record.get("module", "unknown")
                    split = record.get("split", "unknown")
                    key = f"{module}/{split}"
                    try:
                        full_length = int(record["full_length"])
                    except (KeyError, TypeError, ValueError):
                        full_length = None
                    if full_length is not None:
                        diagnostics["full_length_statistics"].setdefault(key, []).append(full_length)
                    for category, count in record.get("collapsed_categories", {}).items():
                        collapsed_counts[category] = collapsed_counts.get(category, 0) + int(count)
    for key, lengths in diagnostics["full_length_statistics"].items():
        lengths.sort()
        n = len(lengths)
        diagnostics["full_length_statistics"][key] = {
            "count": n, "min": lengths[0], "mean": sum(lengths) / n,
            "median": statistics.median(lengths), "p95": lengths[min(n - 1, math.ceil(.95 * n) - 1)],
            "p99": lengths[min(n - 1, math.ceil(.99 * n) - 1)], "max": lengths[-1],
        }
    diagnostics["collapsed_categories"] = collapsed_counts or diagnostics["collapsed_categories"]
    # Summarize prediction completion and visible trace lengths as output data arrives.
    for (model, arm), (result_path, result) in raw_results.items():
        if not result_path or not result:
            continue
        for split in EXP11_SPLITS:
            pred_path = result_path.parent / f"{split}_predictions.csv"
            if not pred_path.is_file():
                continue
            lengths: list[int] = []
            trace_chars: list[int] = []
            trace_nonempty = 0
            structure_scored = structure_valid = 0
            reference_scored = reference_matches = 0
            with pred_path.open(encoding="utf-8", newline="") as handle:
                for record in csv.DictReader(handle):
                    try:
                        lengths.append(int(record.get("output_tokens") or 0))
                        trace = record.get("scratchpad") or ""
                        trace_chars.append(len(trace))
                        trace_nonempty += int(bool(trace))
                        structure = (record.get("trace_structure_valid") or "").lower()
                        if structure in {"true", "false", "1", "0"}:
                            structure_scored += 1
                            structure_valid += int(structure in {"true", "1"})
                        reference = (record.get("trace_reference_exact_match") or "").lower()
                        if reference in {"true", "false", "1", "0"}:
                            reference_scored += 1
                            reference_matches += int(reference in {"true", "1"})
                    except ValueError:
                        continue
            if lengths:
                diagnostics["length_metrics"][f"{model}/{arm}/{split}"] = {
                    "scored_examples": len(lengths),
                    "mean_output_tokens": sum(lengths) / len(lengths),
                    "mean_scratchpad_characters": sum(trace_chars) / len(trace_chars),
                    "mean_scratchpad_tokens": sum(trace_chars) / len(trace_chars),
                    "nonempty_trace_fraction": trace_nonempty / len(lengths),
                    "trace_structure_validity": (
                        structure_valid / structure_scored if structure_scored else None
                    ),
                    "trace_reference_exact_match_rate": (
                        reference_matches / reference_scored if reference_scored else None
                    ),
                }
    compute_matching = {}
    grpo_diagnostics: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        grpo_run_dir = runs_dir / experiment / model / "grpo" / f"seed{seed}"
        grpo_metrics_all = _read_training_metrics(grpo_run_dir / "metrics.csv")
        mixed_rate, mixed_updates = _mean_training_metric(grpo_metrics_all, "group/mixed_rate")
        all_correct_rate, all_correct_updates = _mean_training_metric(
            grpo_metrics_all, "group/all_correct_rate"
        )
        all_wrong_rate, all_wrong_updates = _mean_training_metric(
            grpo_metrics_all, "group/all_wrong_rate"
        )
        update_values = [
            row.get("update") for row in grpo_metrics_all if row.get("update") not in (None, "")
        ]
        grpo_diagnostics[model] = {
            "updates": len(update_values),
            "latest_update": update_values[-1] if update_values else None,
            "informative_group_rate": mixed_rate,
            "informative_group_rate_updates": mixed_updates,
            "mixed_group_rate": mixed_rate,
            "mixed_group_rate_updates": mixed_updates,
            "all_correct_group_rate": all_correct_rate,
            "all_correct_group_rate_updates": all_correct_updates,
            "all_wrong_group_rate": all_wrong_rate,
            "all_wrong_group_rate_updates": all_wrong_updates,
        }
        grpo_metrics = training_metrics_by_arm.get((model, "grpo"), {})
        control_metrics = training_metrics_by_arm.get((model, "continued_sft"), {})
        sft_metrics = training_metrics_by_arm.get((model, "sft"), {})
        try:
            grpo_flops = float(grpo_metrics.get("estimated_flops", ""))
            control_flops = float(control_metrics.get("estimated_flops", ""))
            control_additional_flops = float(
                control_metrics.get("additional_estimated_flops", "")
            )
        except (TypeError, ValueError):
            grpo_flops = control_flops = control_additional_flops = 0.0
        if control_additional_flops <= 0 and control_flops > 0:
            try:
                source_sft_flops = float(sft_metrics.get("estimated_flops", ""))
            except (TypeError, ValueError):
                source_sft_flops = 0.0
            control_additional_flops = max(0.0, control_flops - source_sft_flops)
        ratio = (
            control_additional_flops / grpo_flops
            if grpo_flops > 0 and control_additional_flops > 0 else None
        )
        grpo_step_key = "update"
        control_step_key = "optimizer_step"
        compute_matching[model] = {
            "grpo_step": grpo_metrics.get(grpo_step_key),
            "grpo_estimated_flops": grpo_flops if grpo_flops > 0 else None,
            "continued_sft_step": control_metrics.get(control_step_key),
            "continued_sft_total_estimated_flops": control_flops if control_flops > 0 else None,
            "continued_sft_additional_estimated_flops": (
                control_additional_flops if control_additional_flops > 0 else None
            ),
            "continued_sft_to_grpo_flop_ratio": ratio,
            "status": "measured" if ratio is not None else "pending",
        }

    rlvr_gains = _exp11_rlvr_gains(rows)
    continued_sft_gains = _exp11_continued_sft_gains(rows)
    sample_efficiency: list[dict[str, Any]] = []
    for model in MODELS:
        for arm in ARMS:
            row = next(
                item for item in rows
                if item["model"] == model and item["arm"] == arm and item["split"] == "test"
            )
            # A partial test-split accuracy is real but is not an endpoint for
            # sample-efficiency comparisons, so leave its normalized score blank.
            exp11_test_accuracy = (
                row["exp11_accuracy"] if row["exp11_status"] == "complete" else None
            )
            baseline_per_million_tokens = _efficiency_score(
                row["baseline_accuracy"], row["baseline_logged_tokens"], 1_000_000
            )
            exp11_per_million_tokens = _efficiency_score(
                exp11_test_accuracy, row["logged_tokens"], 1_000_000
            )
            baseline_per_trillion_flops = _efficiency_score(
                row["baseline_accuracy"], row["baseline_estimated_flops"], 1_000_000_000_000
            )
            exp11_per_trillion_flops = _efficiency_score(
                exp11_test_accuracy, row["estimated_flops"], 1_000_000_000_000
            )
            sample_efficiency.append({
                "model": model,
                "arm": arm,
                "split": "test",
                "exp11_status": row["exp11_status"],
                "same_question_identity": row["same_question_identity"],
                "same_flops_estimator": (
                    row["same_flops_estimator"]
                ),
                "baseline_accuracy_pp_per_million_tokens": baseline_per_million_tokens,
                "exp11_accuracy_pp_per_million_tokens": exp11_per_million_tokens,
                "delta_accuracy_pp_per_million_tokens": (
                    exp11_per_million_tokens - baseline_per_million_tokens
                    if row["same_question_identity"]
                    and exp11_per_million_tokens is not None
                    and baseline_per_million_tokens is not None
                    else None
                ),
                "baseline_accuracy_pp_per_trillion_flops": baseline_per_trillion_flops,
                "exp11_accuracy_pp_per_trillion_flops": exp11_per_trillion_flops,
                "delta_accuracy_pp_per_trillion_flops": (
                    exp11_per_trillion_flops - baseline_per_trillion_flops
                    if row["same_question_identity"]
                    and row["same_flops_estimator"]
                    and exp11_per_trillion_flops is not None
                    and baseline_per_trillion_flops is not None
                    else None
                ),
            })

    report = {
        "experiment": experiment, "seed": seed,
        "baseline_snapshot": str(baseline_snapshot),
        "baseline_provenance": {
            "canonical_name": baseline_name,
            "snapshot_path": str(baseline_snapshot),
            "snapshot_sha256": baseline_snapshot_sha256,
            "frozen_at_utc": snapshot.get("frozen_at_utc"),
            "evaluation_wandb_runs": baseline_wandb_runs,
        },
        "row_count": len(rows), "rows": rows, "diagnostics": diagnostics,
        "compute_matching": compute_matching,
        "grpo_diagnostics": grpo_diagnostics,
        "rlvr_gains": rlvr_gains,
        "continued_sft_gains": continued_sft_gains,
        "sample_efficiency": sample_efficiency,
        "statuses": list(EXP11_STATUS),
        "limitations": [
            "Training token counts and FLOPs are read from the latest checkpoint-aligned metric row when available; FLOPs are dense-position estimates, not hardware counters.",
            "Continued SFT uses effective batch 128 as an approximate compute control; its measured FLOP ratio to GRPO is shown when both arm metrics are available.",
            "Accuracy deltas require complete results and identical ordered module/question/target identities in baseline and Experiment 11 prediction CSVs. FLOP deltas and FLOP-efficiency deltas also require matching estimator versions.",
            "The handwritten scratchpad is additional supervised target text in the Experiment 11 SFT data. The current training objective does not separately mask or weight scratchpad tokens, so this is not answer-only supervision with an auxiliary trace loss.",
            "Missing baseline snapshot cells are preserved as missing; unavailable measurements are never treated as zero.",
            "This report covers one seed only and provides no between-seed uncertainty estimate.",
            "The annotation token budget can reduce trace detail; harder or longer examples may therefore receive less detailed supervised traces.",
            "Sample-efficiency ratios are endpoint exact-accuracy percentages normalized by logged tokens or estimated FLOPs; token semantics differ by arm and FLOPs are estimated, so they are descriptive rather than causal comparisons.",
        ],
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(rows[0]) if rows else []
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "intermediate_results.md"
    lines = [
        "# Experiment 11 intermediate results", "", f"Seed: `{seed}`", "",
        f"Rows: {len(rows)} (expected 27: 3 models × 3 arms × 3 splits).",
        "Statuses distinguish missing baseline results, not-run, running, partial, failed, and complete evaluations.",
        "Accuracy deltas are reported only for complete evaluations with identical ordered question identities. FLOP deltas additionally require matching estimator versions.", "",
        "| Model | Arm | Split | Baseline status | Baseline accuracy | Baseline scored | Exp11 status | Exp11 accuracy | Delta (pp) | Exp11 scored | Format valid | Trace structure | Reference trace match | Capped | Mean output tokens | Baseline train tokens | Exp11 train tokens | Additional tokens | Baseline FLOPs | Exp11 FLOPs | Additional FLOPs | Identity match | FLOPs estimator match |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        delta = "Missing" if row["delta_accuracy"] is None else f"{100 * row['delta_accuracy']:+.2f}"
        lines.append(
            f"| {row['model']} | {row['arm']} | {row['split']} | {row['baseline_status']} | "
            f"{_exp11_accuracy(row['baseline_accuracy'])} | {_format_count(row['baseline_examples'])} | "
            f"{row['exp11_status']} | {_exp11_accuracy(row['exp11_accuracy'])} | {delta} | "
            f"{_format_count(row['examples'])} | "
            f"{_exp11_accuracy(row['format_validity'])} | "
            f"{_exp11_accuracy(row['trace_structure_validity'])} | "
            f"{_exp11_accuracy(row['trace_reference_exact_match_rate'])} | "
            f"{_format_count(row['capped_completions'])} | "
            f"{row['mean_output_tokens'] if row['mean_output_tokens'] is not None else 'Missing'} | "
            f"{_format_count(row['baseline_logged_tokens'])} | {_format_count(row['logged_tokens'])} | "
            f"{_format_count(row['additional_tokens'])} | {_format_count(row['baseline_estimated_flops'])} | "
            f"{_format_count(row['estimated_flops'])} | {_format_count(row['additional_flops'])} | "
            f"{'yes' if row['same_question_identity'] else 'unverified'} | "
            f"{'yes' if row['flops_estimator'] == row['baseline_flops_estimator'] and row['flops_estimator'] else 'no/unverified'} |"
        )
    lines.extend([
        "", "## RLVR gains versus SFT", "",
        "A gain is measured only when both Experiment 11 evaluations are complete and their ordered module/question/target identities match. Positive values favor GRPO.",
    ])
    measured_gains = sum(item["status"] == "measured" for item in rlvr_gains)
    lines.extend([
        f"Measured RLVR gains: {measured_gains}/{len(rlvr_gains)} model/split comparisons; pending, failed, and identity-unverified cells do not imply a gain or loss.",
        "", "| Model | Split | Status | GRPO minus SFT (pp) | Status detail |",
        "|---|---|---|---:|---|",
    ])
    for item in rlvr_gains:
        gain = item["grpo_minus_sft_percentage_points"]
        gain_text = f"{gain:+.2f}" if gain is not None else "Missing"
        lines.append(
            f"| {item['model']} | {item['split']} | {item['status']} | {gain_text} | {item['reason']} |"
        )
    measured_control_gains = sum(item["status"] == "measured" for item in continued_sft_gains)
    lines.extend([
        "", "## Continued-SFT gains versus SFT", "",
        f"Measured continued-SFT gains: {measured_control_gains}/{len(continued_sft_gains)} model/split comparisons; pending, failed, and identity-unverified cells do not imply a gain or loss.",
        "", "| Model | Split | Status | Continued SFT minus SFT (pp) | Status detail |",
        "|---|---|---|---:|---|",
    ])
    for item in continued_sft_gains:
        gain = item["continued_sft_minus_sft_percentage_points"]
        gain_text = f"{gain:+.2f}" if gain is not None else "Missing"
        lines.append(
            f"| {item['model']} | {item['split']} | {item['status']} | {gain_text} | {item['reason']} |"
        )
    lines.extend([
        "", "## GRPO informative-group rates", "",
        "Informative groups are mixed-correctness rollout groups: at least one completion is correct and at least one is incorrect. This matches the nonzero binary-reward advantage criterion in the GRPO trainer. Rates are arithmetic means over logged updates.",
        "", "| Model | Updates | Informative / mixed groups | All correct | All wrong |",
        "|---|---:|---:|---:|---:|",
    ])
    for model, metrics in grpo_diagnostics.items():
        informative = metrics["informative_group_rate"]
        all_correct = metrics["all_correct_group_rate"]
        all_wrong = metrics["all_wrong_group_rate"]
        lines.append(
            f"| {model} | {_format_count(metrics['updates'])} | "
            f"{_exp11_accuracy(informative)} ({_format_count(metrics['informative_group_rate_updates'])} updates) | "
            f"{_exp11_accuracy(all_correct)} | {_exp11_accuracy(all_wrong)} |"
        )
    lines.extend([
        "", "## Endpoint sample efficiency", "",
        "Exact ID test accuracy normalized by logged training tokens and estimated FLOPs. FLOP deltas require matching estimator versions. These endpoint ratios are descriptive; they are not learning-curve or causal efficiency estimates.",
        "", "| Model | Arm | Exp11 status | Identity match | FLOPs estimator match | Baseline pp / 1M tokens | Exp11 pp / 1M tokens | Delta | Baseline pp / 1T FLOPs | Exp11 pp / 1T FLOPs | Delta |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in sample_efficiency:
        def efficiency_cell(value: Any) -> str:
            return f"{value:.4g}" if value is not None else "Missing"
        lines.append(
            f"| {item['model']} | {item['arm']} | {item['exp11_status']} | "
            f"{'yes' if item['same_question_identity'] else 'unverified'} | "
            f"{'yes' if item['same_flops_estimator'] else 'no/unverified'} | "
            f"{efficiency_cell(item['baseline_accuracy_pp_per_million_tokens'])} | "
            f"{efficiency_cell(item['exp11_accuracy_pp_per_million_tokens'])} | "
            f"{efficiency_cell(item['delta_accuracy_pp_per_million_tokens'])} | "
            f"{efficiency_cell(item['baseline_accuracy_pp_per_trillion_flops'])} | "
            f"{efficiency_cell(item['exp11_accuracy_pp_per_trillion_flops'])} | "
            f"{efficiency_cell(item['delta_accuracy_pp_per_trillion_flops'])} |"
        )
    lines.extend([
        "", "## Approximate compute matching", "",
        "Continued SFT uses effective batch 128. The ratio below is measured additional continued-SFT estimated FLOPs divided by GRPO estimated FLOPs at the latest checkpoint-aligned metrics; the control is approximate because GRPO rollout lengths vary.",
        "", "| Model | GRPO step | GRPO estimated FLOPs | Continued-SFT step | Continued-SFT additional FLOPs | Continued-SFT / GRPO | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for model, metrics in compute_matching.items():
        ratio = metrics["continued_sft_to_grpo_flop_ratio"]
        ratio_text = f"{ratio:.4f}×" if ratio is not None else "Missing"
        lines.append(
            f"| {model} | {_format_count(metrics['grpo_step'])} | "
            f"{_format_count(metrics['grpo_estimated_flops'])} | "
            f"{_format_count(metrics['continued_sft_step'])} | "
            f"{_format_count(metrics['continued_sft_additional_estimated_flops'])} | {ratio_text} | "
            f"{metrics['status']} |"
        )
    lines.extend([
        "", "## Experiment and W&B provenance", "",
        "Offline runs are identified by their run ID and local directory; a cloud URL appears when W&B provides one.",
        "", "| Model | Arm | Training run | Evaluation run |", "|---|---|---|---|",
    ])
    provenance_rows = {}
    for row in rows:
        provenance_rows.setdefault((row["model"], row["arm"]), row)

    def tracking_cell(provenance: dict[str, Any]) -> str:
        run_id = provenance.get("run_id")
        if not run_id:
            return f"Not started; expected local path `{provenance['local_path']}`"
        run_name = provenance.get("run_name") or "unnamed"
        url = provenance.get("url")
        label = f"[{run_name}]({url})" if url else run_name
        mode = provenance.get("mode") or "unknown mode"
        sync_status = provenance.get("sync_status") or "unknown sync"
        local_path = provenance.get("local_path") or ""
        return f"{label} (`{run_id}`; {mode}/{sync_status}; `{local_path}`)"

    for (model, arm), row in sorted(provenance_rows.items()):
        lines.append(
            f"| {model} | {arm} | {tracking_cell(row['training_wandb'])} | "
            f"{tracking_cell(row['evaluation_wandb'])} |"
        )
    baseline_meta = report["baseline_provenance"]
    lines.extend([
        "", "## Frozen Experiment 10 baseline provenance", "",
        f"- Snapshot: `{baseline_meta['snapshot_path']}`.",
        f"- Snapshot SHA-256: `{baseline_meta['snapshot_sha256'] or 'unavailable'}`.",
        f"- Frozen at UTC: `{baseline_meta['frozen_at_utc'] or 'unavailable'}`.",
        f"- Canonical baseline: `{baseline_meta['canonical_name']}`.",
        "", "| Model | Arm | W&B run ID | Mode / sync | Status | Checkpoint selector / step | Local metadata path |",
        "|---|---|---|---|---|---|---|",
    ])
    for model, arm_map in sorted(baseline_meta["evaluation_wandb_runs"].items()):
        for arm, provenance in sorted(arm_map.items()):
            checkpoint = provenance.get("checkpoint_selector") or "Missing"
            if provenance.get("checkpoint_step") is not None:
                checkpoint += f" / {provenance['checkpoint_step']}"
            lines.append(
                f"| {model} | {arm} | {provenance.get('id', 'Missing')} | "
                f"{provenance.get('mode', 'unknown')} / {provenance.get('sync_status', 'unknown')} | "
                f"{provenance.get('run_status', 'unknown')} | {checkpoint} | "
                f"`{provenance.get('path', 'Missing')}` |"
            )
    lines.extend([
        "", "## Evaluation accuracy and format by module", "",
        "| Model | Arm | Split | Module | Examples | Accuracy | Format valid | Trace structure | Reference trace match |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        for module, metrics in sorted(row["module_metrics"].items()):
            lines.append(
                f"| {row['model']} | {row['arm']} | {row['split']} | {module} | "
                f"{_format_count(metrics.get('examples'))} | {_exp11_accuracy(metrics.get('accuracy'))} | "
                f"{_exp11_accuracy(metrics.get('format_validity'))} | "
                f"{_exp11_accuracy(metrics.get('trace_structure_validity'))} | "
                f"{_exp11_accuracy(metrics.get('trace_reference_exact_match_rate'))} |"
            )
    lines.extend([
        "", "## Format, trace, length, and collapse diagnostics", "",
        f"Annotation protocol: `{diagnostics['protocol']}`; annotation hash: `{diagnostics['annotation_sha256'] or 'unavailable'}`.",
        f"Source bundle: `{diagnostics['annotation_provenance'].get('source_bundle') or 'unavailable'}`; source manifest SHA-256: `{diagnostics['annotation_provenance'].get('source_manifest_sha256') or 'unavailable'}`; source split SHA-256: `{diagnostics['annotation_provenance'].get('source_split_sha256') or 'unavailable'}`; source tokenizer SHA-256: `{diagnostics['annotation_provenance'].get('source_tokenizer_sha256') or 'unavailable'}`; annotation tokenizer SHA-256: `{diagnostics['annotation_provenance'].get('tokenizer_sha256') or 'unavailable'}`.",
        "", "| Module / split | Retained-token count | Min | Mean | Median | P95 | P99 | Max | Reduced examples | Reduced fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, stats in sorted(diagnostics["module_split_statistics"].items()):
        lines.append(
            f"| {group} | {_format_count(stats.get('count'))} | {_format_count(stats.get('min'))} | "
            f"{_format_count(stats.get('mean'))} | {_format_count(stats.get('median'))} | {_format_count(stats.get('p95'))} | {_format_count(stats.get('p99'))} | "
            f"{_format_count(stats.get('max'))} | {_format_count(stats.get('reduced_count'))} | "
            f"{100 * float(stats.get('reduced_fraction') or 0):.2f}% |"
        )
    lines.extend([
        "", "### Full untruncated annotation lengths", "",
        "These source lengths are measured before trace reduction; retained-token lengths above describe the actual training target budget.",
        "", "| Module / split | Count | Min | Mean | Median | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group, stats in sorted(diagnostics["full_length_statistics"].items()):
        lines.append(
            f"| {group} | {_format_count(stats.get('count'))} | {_format_count(stats.get('min'))} | "
            f"{stats.get('mean', 0):.2f} | {_format_count(stats.get('median'))} | "
            f"{_format_count(stats.get('p95'))} | {_format_count(stats.get('p99'))} | {_format_count(stats.get('max'))} |"
        )
    lines.extend([
        "", "Collapse operations: " + (
            ", ".join(f"{name}={count:,}" for name, count in sorted(diagnostics["collapsed_categories"].items()))
            if diagnostics["collapsed_categories"] else "unavailable until annotated rows are present"
        ) + ".",
        "Trace length metrics by model/arm/split are in `comparison.json`; they populate as prediction CSVs arrive.",
        "The evaluation results expose format validity, capped completions, output tokens, and prediction CSVs include the generated scratchpad for trace inspection.",
        "",
        "## Limitations", "",
        *[f"- {item}" for item in report["limitations"]],
        "- Additional training tokens and FLOPs are compared only when both Experiment 10 and Experiment 11 checkpoint-aligned metrics exist; unavailable counters remain blank.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    chart_rows = [
        {**row, "accuracy": row.get("exp11_accuracy"), "parameters": row.get("parameters")}
        for row in rows
    ]
    figures = [
        *_exp11_charts(rows, output_dir),
        *_figure_scaling(chart_rows, output_dir),
        *_figure_arm_bars(chart_rows, output_dir),
        *_exp11_breakdown_charts(chart_rows, output_dir),
        *_figure_curves(runs_dir, experiment, output_dir, seed),
        *_figure_ood(chart_rows, output_dir),
        *_figure_compute(chart_rows, runs_dir, experiment, output_dir, seed),
        *_exp11_grpo_sample_efficiency_chart(
            runs_dir=runs_dir, experiment=experiment, output_dir=output_dir, seed=seed
        ),
    ]
    return report, [markdown_path, report_path, csv_path, *figures]
