"""Collect measured local runs and render comparable figures and tables."""

from __future__ import annotations

import csv
import json
import math
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
        if arm == "grpo" and selected.get("flops_estimator_version") != "dense_position_v2":
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
        axis.set_xlabel("Estimated training FLOPs (dense 2N/6N token-position model)")
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
            "- SFT token counts include processed input positions; GRPO token counts include generated completion tokens, so token efficiency is descriptive within each arm, not a strictly matched cross-arm measure.",
            "- FLOP and token efficiency are descriptive proxies tied to the logged counters; they are not hardware energy or latency measurements.",
            "- Accuracy differences are descriptive for this protocol and seed; they do not establish a general scaling or RLVR effect beyond these runs.",
            "",
        ]
    )
    return "\n".join(lines)


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
        "compute_estimate": "SFT uses 6PN; GRPO counts no-cache rollout prefix positions and policy/reference token positions with dense 2N/6N multipliers",
        "limitations": [
            "Compute uses a dense-transformer token-position approximation; it is not hardware-counter FLOPs.",
            "GRPO compute points are omitted when a run lacks the dense_position_v2 estimator marker.",
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
