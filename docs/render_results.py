"""Render README figures from the versioned snapshot, without experiment tracking.

Run from the repository root: uv run --locked python docs/render_results.py
No local training artifacts, credentials, or network access are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("1m", "10m", "100m")
SPLITS = ("test", "number_length_ood", "expression_length_ood")
TITLES = ("Test", "Number-length OOD", "Expression-length OOD")
BLUE, ORANGE, TEAL = "#1f77b4", "#ff7f0e", "#16857b"
INK, MUTED = "#172b42", "#586779"


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e3e8ef", linewidth=0.8)
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    for name in ("bottom", "left"):
        axis.spines[name].set_color("#cbd4df")
    axis.tick_params(colors=MUTED, length=0, pad=8)


def save_figure(fig: plt.Figure, stem: str) -> None:
    output = ROOT / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    svg_path = output / f"{stem}.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    # Matplotlib emits trailing spaces in path data; keep versioned SVG diffs clean.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(output / f"{stem}.png", dpi=200)
    plt.close(fig)
    print(f"Rendered docs/assets/{stem}.{{svg,png}}")


def main() -> None:
    snapshot = json.loads((ROOT / "docs/results/reasoning-v3-seed18.json").read_text())
    training = snapshot.get("training", {})
    rows = snapshot["measurements"]
    lookup = {(row["model"], row["arm"], row["split"]): row for row in rows}
    expected = {
        (model, arm, split)
        for model in MODELS
        for arm in ("sft", "grpo", "continued_sft")
        for split in SPLITS
    }
    if set(lookup) != expected or len(rows) != len(expected):
        raise ValueError("The documentation snapshot must contain all 27 unique comparisons")
    for row in rows:
        if row["examples"] != snapshot["evaluation"]["examples_per_split"][row["split"]]:
            raise ValueError("Evaluation population differs from the documented split")
        if not all(0 <= row[key] <= 1 for key in ("baseline_accuracy", "reasoning_v3_accuracy")):
            raise ValueError("Accuracy must be a measured fraction in [0, 1]")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "tinymath-readme-seed18",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.23, top=0.68, wspace=0.13)
    fig.text(0.06, 0.94, "Learning arithmetic at three model sizes", fontsize=20, weight="bold")
    fig.text(0.06, 0.875, "SFT comparison  ·  fixed evaluation questions  ·  seed 18", color=MUTED)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", linestyle="--", label="Baseline SFT"),
            Line2D([], [], color=ORANGE, marker="o", linewidth=2.4, label="Reasoning-v3 SFT"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.054, 0.84),
        frameon=False,
        ncol=2,
    )
    parameters = [snapshot["model_parameters"][model] for model in MODELS]
    for axis, split, title in zip(axes, SPLITS, TITLES):
        style_axis(axis)
        count = snapshot["evaluation"]["examples_per_split"][split]
        axis.set_title(f"{title}\n{count:,} questions", loc="left", fontsize=11, pad=15)
        for field, color, linestyle, offset in (
            ("baseline_accuracy", BLUE, "--", -17),
            ("reasoning_v3_accuracy", ORANGE, "-", 10),
        ):
            values = [100 * lookup[model, "sft", split][field] for model in MODELS]
            axis.plot(
                parameters,
                values,
                color=color,
                linestyle=linestyle,
                marker="o",
                linewidth=2.2,
                markersize=6,
            )
            for x, y in zip(parameters, values):
                label_offset = -8 if field == "baseline_accuracy" and y < 10 else offset
                axis.annotate(
                    f"{y:.2f}",
                    (x, y),
                    xytext=(0, label_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=color,
                )
        axis.set_xscale("log")
        axis.set_xlim(parameters[0] / 1.6, parameters[-1] * 1.6)
        axis.set_xticks(parameters, ("1M", "10M", "100M"))
        axis.minorticks_off()
        axis.set_ylim(-3, 105)
        axis.set_yticks((0, 25, 50, 75, 100))
        axis.set_xlabel("Model parameters", labelpad=10)
    axes[0].set_ylabel("Final-answer accuracy (%)", labelpad=10)
    fig.text(
        0.06,
        0.07,
        "Reasoning-v3 includes scratchpads, subtask training examples, and exact arithmetic on eligible compound tasks.",
        fontsize=9,
        color=MUTED,
    )
    save_figure(fig, "reasoning-v3-scaling")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.23, top=0.68, wspace=0.13)
    fig.text(0.06, 0.94, "What changes after SFT?", fontsize=20, weight="bold")
    fig.text(
        0.06,
        0.875,
        "Reasoning-v3  ·  accuracy change from each model's SFT checkpoint  ·  seed 18",
        color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(facecolor=ORANGE, label="GRPO · 10,000 updates"),
            Patch(
                facecolor=TEAL,
                label=training.get(
                    "continued_sft_display_label",
                    "Continued SFT · 100,000 additional steps",
                ),
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.054, 0.84),
        frameon=False,
        ncol=2,
    )
    x = np.arange(len(MODELS))
    width = 0.32
    for axis, split, title in zip(axes, SPLITS, TITLES):
        style_axis(axis)
        axis.set_title(title, loc="left", fontsize=11, pad=15)
        axis.axhline(0, color=INK, linewidth=1)
        for arm, color, shift in (("grpo", ORANGE, -width / 2), ("continued_sft", TEAL, width / 2)):
            values = [
                100
                * (
                    lookup[model, arm, split]["reasoning_v3_accuracy"]
                    - lookup[model, "sft", split]["reasoning_v3_accuracy"]
                )
                for model in MODELS
            ]
            axis.bar(x + shift, values, width=width, color=color, zorder=3)
            for index, (position, value) in enumerate(zip(x + shift, values)):
                label_offset = 6 if value >= 0 else -7
                grpo_gain = 100 * (
                    lookup[MODELS[index], "grpo", split]["reasoning_v3_accuracy"]
                    - lookup[MODELS[index], "sft", split]["reasoning_v3_accuracy"]
                )
                if arm == "continued_sft" and value * grpo_gain > 0 and abs(value - grpo_gain) < 1:
                    label_offset = 18 if value > 0 else -19
                axis.annotate(
                    f"{value:+.2f}",
                    (position, value),
                    xytext=(0, label_offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=9,
                    color=INK,
                )
        axis.set_xticks(x, ("1M", "10M", "100M"))
        axis.set_xlabel("Model size", labelpad=10)
        axis.set_ylim(-5, 7)
        axis.set_yticks((-4, -2, 0, 2, 4, 6))
    axes[0].set_ylabel("Accuracy change (percentage points)", labelpad=10)
    fig.text(
        0.06,
        0.07,
        training.get(
            "continued_sft_plot_note",
            "Additional compute budgets differ. These are descriptive changes; a single seed gives no estimate of seed variability.",
        ),
        fontsize=9,
        color=MUTED,
    )
    save_figure(fig, "reasoning-v3-training-gains")


if __name__ == "__main__":
    main()
