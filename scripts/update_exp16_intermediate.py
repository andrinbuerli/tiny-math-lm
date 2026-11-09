"""Refresh live Experiment 16 Markdown and plots from durable progress rows."""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from tiny_math_lm.config import workspace_path

LOGGER = logging.getLogger(__name__)
EXPERIMENT_DIR = "exp16_qwen3_base_reasoning_v3"
MODEL_DIR = "qwen3_0_6b_base"
EXPECTED = {"test": 50_000, "number_length_ood": 5_000, "expression_length_ood": 2_000}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue  # Ignore a partially written final line; it will be read next refresh.
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    summary: dict[str, Any] = {"rows": count}
    for field in (
        "correct",
        "format_valid",
        "full_success",
        "trace_valid",
        "capped",
        "executor_failure",
    ):
        summary[field] = sum(row.get(field) is True for row in rows) / count if count else None
    summary["mean_output_tokens"] = (
        sum(float(row.get("output_tokens", 0) or 0) for row in rows) / count if count else None
    )
    summary["executor_calls"] = sum(int(row.get("call_count", 0) or 0) for row in rows)
    summary["policy_tokens"] = sum(int(row.get("policy_tokens", 0) or 0) for row in rows)
    summary["executor_tokens"] = sum(int(row.get("executor_tokens", 0) or 0) for row in rows)
    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    operators: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        modules[str(row.get("module", "unknown"))].append(row)
        digits[str(row.get("digit_width", "unknown"))].append(row)
        operators[str(row.get("operator_count", "unknown"))].append(row)

    def grouped_accuracy(groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        def category_key(value: str) -> tuple[int, float | str]:
            try:
                return 0, float(value)
            except ValueError:
                return 1, value

        return {
            key: {
                "rows": len(values),
                "accuracy": sum(row.get("correct") is True for row in values) / len(values),
            }
            for key, values in sorted(groups.items(), key=lambda item: category_key(item[0]))
        }

    summary["modules"] = {
        name: {
            "rows": len(values),
            "accuracy": sum(row.get("correct") is True for row in values) / len(values),
        }
        for name, values in sorted(modules.items())
    }
    summary["by_digit_width"] = grouped_accuracy(digits)
    summary["by_operator_count"] = grouped_accuracy(operators)
    return summary


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def _pilot_results(runs_root: Path, seed: int) -> list[dict[str, Any]]:
    root = runs_root / EXPERIMENT_DIR / MODEL_DIR
    results: list[dict[str, Any]] = []
    for arm in ("sft", "grpo"):
        arm_root = root / arm
        if not arm_root.is_dir():
            continue
        for phase_root in sorted(arm_root.iterdir()):
            if phase_root.name not in {"pilot", "smoke", "optimization_pilot"}:
                continue
            run_dir = phase_root / f"seed{seed}"
            if not run_dir.is_dir():
                continue
            rows: list[dict[str, str]] = []
            metrics_path = run_dir / "metrics.csv"
            if metrics_path.is_file():
                with metrics_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            tracking_path = run_dir / "wandb_run.json"
            tracking = (
                json.loads(tracking_path.read_text(encoding="utf-8"))
                if tracking_path.is_file()
                else {}
            )
            latest = rows[-1] if rows else {}
            results.append(
                {
                    "arm": arm.upper(),
                    "phase": phase_root.name,
                    "status": str(tracking.get("run_status", "unknown")),
                    "sync_status": str(tracking.get("sync_status", "unknown")),
                    "updates": latest.get("update", latest.get("optimizer_step")),
                    "wall_seconds": latest.get("wall_seconds"),
                    "tokens_per_second": latest.get("tokens_per_second"),
                    "peak_gpu_memory_bytes": latest.get("peak_gpu_memory_bytes"),
                    "wandb_id": tracking.get("id"),
                    "wandb_url": tracking.get("url"),
                    "has_metrics": bool(rows),
                }
            )
    return results


def _training_results(runs_root: Path, seed: int) -> list[dict[str, Any]]:
    """Read durable metrics from main and control training runs, tolerating a live CSV tail."""
    root = runs_root / EXPERIMENT_DIR / MODEL_DIR
    results: list[dict[str, Any]] = []
    for arm_root in sorted(root.iterdir() if root.is_dir() else []):
        if not arm_root.is_dir() or arm_root.name not in {"sft", "grpo", "continued_sft"}:
            continue
        for phase_root in sorted(arm_root.iterdir()):
            if not phase_root.is_dir() or phase_root.name in {"pilot", "smoke", "optimization_pilot"}:
                continue
            run_dir = phase_root / f"seed{seed}"
            metrics_path = run_dir / "metrics.csv"
            if not metrics_path.is_file():
                continue
            rows: list[dict[str, str]] = []
            try:
                with metrics_path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("optimizer_step") or row.get("update"):
                            rows.append(row)
            except (OSError, csv.Error):
                LOGGER.exception("Could not read live training metrics at %s", metrics_path)
            if not rows:
                continue
            tracking_path = run_dir / "wandb_run.json"
            try:
                tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                tracking = {}
            latest = dict(rows[-1])
            validation_rows = [
                row for row in rows if _as_float(row.get("validation/completion_loss")) is not None
            ]
            if validation_rows:
                latest_validation = validation_rows[-1]
                latest["validation/completion_loss"] = latest_validation[
                    "validation/completion_loss"
                ]
                latest["validation_step"] = latest_validation.get(
                    "optimizer_step", latest_validation.get("update", "—")
                )
            greedy_rows = [
                row for row in rows
                if _as_float(row.get("validation/greedy_accuracy")) is not None
            ]
            if greedy_rows:
                latest_greedy = greedy_rows[-1]
                latest["validation/greedy_accuracy"] = latest_greedy[
                    "validation/greedy_accuracy"
                ]
                latest["validation/greedy_accuracy_examples"] = latest_greedy.get(
                    "validation/greedy_accuracy_examples"
                )
                latest["greedy_accuracy_step"] = latest_greedy.get(
                    "optimizer_step", latest_greedy.get("update", "—")
                )
            results.append(
                {
                    "arm": arm_root.name.upper(),
                    "phase": phase_root.name,
                    "status": str(tracking.get("run_status", "running")),
                    "sync_status": str(tracking.get("sync_status", "unknown")),
                    "metrics": rows,
                    "latest": latest,
                    "wandb_id": tracking.get("id"),
                    "wandb_url": tracking.get("url"),
                }
            )
    return results


def _qualitative_results(runs_root: Path, seed: int) -> list[dict[str, Any]]:
    run_dir = (
        runs_root / EXPERIMENT_DIR / MODEL_DIR / "sft" / "main" / f"seed{seed}"
    )
    snapshots = sorted((run_dir / "qualitative").glob("step_*.jsonl"))
    return _read_jsonl(snapshots[-1]) if snapshots else []


def _markdown_cell(value: Any) -> str:
    return str(value or "—").replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def _render_markdown(
    summaries: dict[str, dict[str, Any]],
    updated: str,
    pilot_results: list[dict[str, Any]] | None = None,
    training_results: list[dict[str, Any]] | None = None,
    qualitative_results: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# Experiment 16 intermediate results",
        "",
        f"Last refreshed: {updated}",
        "",
        "> Live, incomplete results. Accuracy and other rates use rows durably written so far; do not treat them as final estimates.",
        "",
        "## Evaluation progress",
        "",
        "| Arm | Split | Format | Rows | Expected | Accuracy | Format valid | Protocol success | Trace valid | Capped | Mean output tokens | Executor calls | Executor failures | Policy tokens | Executor tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for relative, summary in sorted(summaries.items()):
        parts = Path(relative).parts
        try:
            phase_index = parts.index("evaluation") + 1
            arm = parts[phase_index]
            checkpoint = parts[phase_index + 1]
            filename = Path(relative).name.removesuffix(".jsonl")
            split = next(name for name in EXPECTED if filename.startswith(f"{name}_"))
            format_name = filename.removeprefix(f"{split}_")
        except (ValueError, IndexError):
            arm, checkpoint, split, format_name = "unknown", "unknown", "unknown", "unknown"
        expected = EXPECTED.get(split)
        label = f"{arm}/{checkpoint}"
        lines.append(
            f"| {label} | {split} | {format_name} | {summary['rows']} | {expected or '—'} | "
            f"{_format(summary['correct'], percent=True)} | {_format(summary['format_valid'], percent=True)} | "
            f"{_format(summary['full_success'], percent=True)} | {_format(summary['trace_valid'], percent=True)} | "
            f"{_format(summary['capped'], percent=True)} | {_format(summary['mean_output_tokens'])} | "
            f"{summary['executor_calls']} | {_format(summary.get('executor_failure'), percent=True)} | "
            f"{summary['policy_tokens']} | {summary['executor_tokens']} |"
        )
        if summary["modules"]:
            lines.extend(
                [
                    "",
                    f"### {label} · {split} · {format_name}",
                    "",
                    "| Module | Rows | Accuracy |",
                    "| --- | ---: | ---: |",
                ]
            )
            for module, metrics in summary["modules"].items():
                lines.append(
                    f"| {module} | {metrics['rows']} | {_format(metrics['accuracy'], percent=True)} |"
                )
            lines.append("")
        for field, heading in (
            ("by_digit_width", "Digit width"),
            ("by_operator_count", "Operator count"),
        ):
            groups = summary.get(field, {})
            if groups:
                lines.extend(
                    [
                        f"### {label} · {split} · {format_name} · {heading.lower()}",
                        "",
                        f"| {heading} | Rows | Accuracy |",
                        "| ---: | ---: | ---: |",
                    ]
                )
                lines.extend(
                    f"| {category} | {metrics['rows']} | {_format(metrics['accuracy'], percent=True)} |"
                    for category, metrics in groups.items()
                )
                lines.append("")
    if not summaries:
        lines.extend(["", "No evaluation progress JSONL files are available yet."])
    lines.extend(["", "## Tracked pilot results", ""])
    pilot_results = pilot_results or []
    if pilot_results:
        lines.extend(
            [
                "| Arm | Phase | Status | Updates/steps | Wall seconds | Tokens/s | Peak GPU GB | W&B | Sync |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for result in pilot_results:
            link = (
                f"[{result['wandb_id']}]({result['wandb_url']})"
                if result.get("wandb_id") and result.get("wandb_url")
                else "—"
            )
            peak_bytes = result.get("peak_gpu_memory_bytes")
            try:
                peak_gb = f"{float(peak_bytes) / 1_000_000_000:.2f}"
            except (TypeError, ValueError):
                peak_gb = "—"
            lines.append(
                f"| {result['arm']} | {result['phase']} | {result['status']} | "
                f"{result.get('updates') or '—'} | {result.get('wall_seconds') or '—'} | "
                f"{result.get('tokens_per_second') or '—'} | {peak_gb} | {link} | "
                f"{result['sync_status']} |"
            )
        incomplete = [result for result in pilot_results if not result.get("has_metrics")]
        if incomplete:
            lines.extend(
                [
                    "",
                    "Failed or interrupted pilots without a completed optimizer update have no metrics row; inspect their linked W&B logs for the failure details.",
                ]
            )
    else:
        lines.append("No pilot metrics are available yet.")
    lines.extend(["", "## Main training progress", ""])
    training_results = training_results or []
    if training_results:
        lines.extend(
            [
                "| Arm | Phase | Status | Updates | Epoch | Completion loss | Last validation loss (step) | Greedy exact match (n, step) | Tokens seen | Tokens/s | Wall seconds | Peak GPU GB | W&B | Sync |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for result in training_results:
            latest = result["latest"]
            link = (
                f"[{result['wandb_id']}]({result['wandb_url']})"
                if result.get("wandb_id") and result.get("wandb_url")
                else "—"
            )
            try:
                peak_gb = f"{float(latest.get('peak_gpu_memory_bytes', '')) / 1_000_000_000:.2f}"
            except (TypeError, ValueError):
                peak_gb = "—"
            lines.append(
                f"| {result['arm']} | {result['phase']} | {result['status']} | "
                f"{latest.get('optimizer_step', latest.get('update', '—'))} | "
                f"{_format(_as_float(latest.get('epoch')))} | "
                f"{_format(_as_float(latest.get('train/completion_loss')))} | "
                f"{_format(_as_float(latest.get('validation/completion_loss')))}"
                f" @ {latest.get('validation_step', '—')} | "
                f"{_format(_as_float(latest.get('validation/greedy_accuracy')), percent=True)} "
                f"({latest.get('validation/greedy_accuracy_examples', '—') or '—'}, "
                f"{latest.get('greedy_accuracy_step', '—')}) | "
                f"{latest.get('tokens_seen', '—')} | "
                f"{_format(_as_float(latest.get('tokens_per_second')))} | "
                f"{_format(_as_float(latest.get('wall_seconds')))} | {peak_gb} | {link} | "
                f"{result['sync_status']} |"
            )
    else:
        lines.append("No main-arm training metrics are available yet.")
    qualitative_results = qualitative_results or []
    lines.extend(["", "## Qualitative validation examples", ""])
    if qualitative_results:
        sample_step = qualitative_results[0].get("optimizer_step", "—")
        lines.extend(
            [
                f"Deterministic parent-only validation samples at SFT optimizer step {sample_step}; these are illustrative examples, not an accuracy estimate.",
                "",
            ]
        )
        for index, row in enumerate(qualitative_results, start=1):
            lines.extend(
                [
                    f"### {index}. {_markdown_cell(row.get('module'))}",
                    "",
                    f"- Question: {_markdown_cell(row.get('question'))}",
                    f"- Target: `{_markdown_cell(row.get('target'))}`; prediction: `{_markdown_cell(row.get('prediction'))}`",
                    f"- Format valid: `{row.get('format_valid')}`; trace valid: `{row.get('trace_valid')}`; correct: `{row.get('correct')}`; full success: `{row.get('full_success')}`",
                    "",
                    "Generated completion:",
                    "",
                    "```text",
                    str(row.get("completion", "")),
                    "```",
                    "",
                ]
            )
    else:
        lines.append("No qualitative SFT snapshot has been written yet.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Only completed rows are summarized. The fixed evaluation population sizes are test 50,000, number-length OOD 5,000, and expression-length OOD 2,000. This is one seed; these interim rates do not estimate between-seed uncertainty. Pretrained-data contamination cannot be excluded, and historical TinyMath comparisons differ in tokenizer, architecture, model size, and pretraining.",
            "",
            "Plots: `intermediate_accuracy.svg` and `intermediate_progress.svg` (PNG versions are also provided).",
            "",
        ]
    )
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _write_figures(output: Path, series: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for name, rows in sorted(series.items()):
        if not rows:
            continue
        correct = [int(row.get("correct") is True) for row in rows]
        window = max(1, min(500, len(correct) // 100 or 1))
        x, y = [], []
        for index in range(window, len(correct) + 1, window):
            x.append(index)
            y.append(sum(correct[index - window : index]) / window)
        if len(correct) % window:
            x.append(len(correct))
            y.append(sum(correct[len(correct) - len(correct) % window :]) / (len(correct) % window))
        axis.plot(x, y, label=name)
        plotted = True
    axis.set_xlabel("Evaluated rows")
    axis.set_ylabel("Accuracy in recent window")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize="small")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"intermediate_accuracy.{ext}", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    names = sorted(series)
    values = [len(series[name]) for name in names]
    bars = axis.bar(names or ["waiting"], values or [0])
    for bar, count in zip(bars, values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    axis.set_ylabel("Durably written rows")
    axis.set_title("Evaluation progress")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"intermediate_progress.{ext}", dpi=160)
    plt.close(fig)

    for field, filename, label in (
        ("digit_width", "intermediate_by_digit_width", "Digit width"),
        ("operator_count", "intermediate_by_operator_count", "Operator count"),
    ):
        fig, axis = plt.subplots(figsize=(9, 5))
        plotted = False
        for name, rows in sorted(series.items()):
            grouped: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                category = row.get(field)
                if category is not None:
                    grouped[str(category)].append(row.get("correct") is True)
            try:
                categories = sorted(grouped, key=float)
            except ValueError:
                categories = sorted(grouped)
            if categories:
                try:
                    x_values = [float(category) for category in categories]
                    axis.set_xticks(x_values, categories)
                except ValueError:
                    x_values = categories
                axis.plot(
                    x_values,
                    [sum(grouped[key]) / len(grouped[key]) for key in categories],
                    marker="o",
                    label=name,
                )
                plotted = True
        axis.set_xlabel(label)
        axis.set_ylabel("Mathematical accuracy")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize="small")
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(output / f"{filename}.{ext}", dpi=160)
        plt.close(fig)


def _write_training_figure(output: Path, training_results: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5))
    plotted = False
    for result in training_results:
        rows = result["metrics"]
        steps: list[int] = []
        train_losses: list[float] = []
        validation_steps: list[int] = []
        validation_losses: list[float] = []
        for row in rows:
            step = _as_float(row.get("optimizer_step", row.get("update")))
            train_loss = _as_float(row.get("train/completion_loss", row.get("loss")))
            validation_loss = _as_float(row.get("validation/completion_loss"))
            if step is None:
                continue
            if train_loss is not None:
                steps.append(int(step))
                train_losses.append(train_loss)
            if validation_loss is not None:
                validation_steps.append(int(step))
                validation_losses.append(validation_loss)
        label = f"{result['arm']} {result['phase']}"
        if steps:
            axis.plot(steps, train_losses, label=f"{label} train")
            plotted = True
        if validation_steps:
            axis.plot(validation_steps, validation_losses, marker="o", label=f"{label} validation")
            plotted = True
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Completion loss")
    axis.set_title("Training progress")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize="small")
    else:
        axis.text(0.5, 0.5, "Waiting for training metrics", ha="center", va="center", transform=axis.transAxes)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"intermediate_training_loss.{ext}", dpi=160)
    plt.close(fig)


def refresh(*, runs_root: Path, output_dir: Path, seed: int) -> int:
    root = runs_root / EXPERIMENT_DIR / MODEL_DIR
    progress_files = sorted(root.glob(f"evaluation/*/*/seed{seed}/progress/*.jsonl"))
    summaries: dict[str, dict[str, Any]] = {}
    series: dict[str, list[dict[str, Any]]] = {}
    for path in progress_files:
        rows = _read_jsonl(path)
        relative = str(path.relative_to(root))
        summaries[relative] = _summarize(rows)
        series[relative.replace("/progress/", "/").removesuffix(".jsonl")] = rows
    _write_figures(output_dir, series)
    updated = datetime.now(UTC).isoformat(timespec="seconds")
    pilot_results = _pilot_results(runs_root, seed)
    training_results = _training_results(runs_root, seed)
    qualitative_results = _qualitative_results(runs_root, seed)
    _write_training_figure(output_dir, training_results)
    _atomic_text(
        output_dir / "intermediate_results.md",
        _render_markdown(
            summaries,
            updated,
            pilot_results,
            training_results,
            qualitative_results,
        ),
    )
    return sum(summary["rows"] for summary in summaries.values())


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    output_dir = workspace_path(str(config.reporting.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_yaml(config, resolve=True)
    _atomic_text(output_dir / "intermediate_resolved_config.yaml", resolved)
    runs_root = workspace_path("runs")
    seed = int(config.seed)
    refresh_seconds = max(10, int(config.reporting.intermediate_refresh_seconds))
    orchestrator_pid = config.reporting.get("intermediate_orchestrator_pid")
    while True:
        rows = refresh(runs_root=runs_root, output_dir=output_dir, seed=seed)
        LOGGER.info("Updated live Experiment 16 report with %d evaluation rows", rows)
        if not orchestrator_pid or not _pid_alive(int(orchestrator_pid)):
            break
        time.sleep(refresh_seconds)


if __name__ == "__main__":
    main()
