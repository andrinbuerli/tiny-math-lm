"""Hydra orchestrator for the ordered, resumable Experiment 16 stages."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from tinymath.config import workspace_path

LOGGER = logging.getLogger(__name__)


def _run(config: DictConfig, script: str, *overrides: str) -> None:
    uv = shutil.which("uv") or "/workspace/.tools/uv"
    command = [
        uv,
        "run",
        "--locked",
        "python",
        f"scripts/{script}.py",
        f"experiment={config.experiment.name}",
        *overrides,
    ]
    LOGGER.info("Running Experiment 16 stage: %s", " ".join(command[3:]))
    subprocess.run(command, cwd="/workspace", check=True)


def _require_checkpoint(path: Path, label: str) -> None:
    if not (path / "config.json").is_file() or not (path / "experiment_identity.json").is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing or incomplete: {path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if str(config.model.get("backend", "tinymath")) != "huggingface":
        raise ValueError("Experiment 16 orchestration requires the Hugging Face backend")
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 16 orchestration requires the admitted RTX 4090")
    seed = int(config.seed)
    root = workspace_path("runs", str(config.experiment.name), str(config.model.name))
    sft_run = root / "sft" / "main" / f"seed{seed}"
    selected_sft = sft_run / "best"
    grpo_run = root / "grpo" / "main" / f"seed{seed}"
    final_grpo = grpo_run / "final"
    continued_run = root / "continued_sft" / "continued_sft" / f"seed{seed}"
    final_control = continued_run / "final"

    _run(
        config,
        "train_sft",
        "experiment.phase=pilot",
        "trainer.name=sft",
        "trainer.max_steps=2",
        "trainer.save_every=1",
        "trainer.eval_every=500",
    )
    _run(
        config,
        "train_rlvr",
        "experiment.phase=pilot",
        "rlvr.updates=1",
        "rlvr.save_every=1",
    )
    _run(config, "evaluate", "experiment.phase=base_zero_shot", "checkpoint=null")

    _run(config, "train_sft", "experiment.phase=main", "trainer.name=sft")
    _require_checkpoint(selected_sft, "Selected SFT")
    _run(
        config,
        "evaluate",
        "experiment.phase=sft_evaluation",
        f"checkpoint={selected_sft.relative_to('/workspace')}",
    )

    _run(
        config,
        "train_rlvr",
        "experiment.phase=main",
        f"init_checkpoint={selected_sft.relative_to('/workspace')}",
    )
    _require_checkpoint(final_grpo, "Final GRPO")
    _run(
        config,
        "evaluate",
        "experiment.phase=grpo_evaluation",
        f"checkpoint={final_grpo.relative_to('/workspace')}",
    )
    grpo_state = torch.load(grpo_run / "latest.pt", map_location="cpu", weights_only=False)
    target_flops = int(grpo_state["estimated_flops"])
    if target_flops <= 0:
        raise ValueError("Completed GRPO checkpoint has no positive FLOP estimate")

    _run(
        config,
        "train_sft",
        "experiment.phase=continued_sft",
        "trainer.name=continued_sft",
        f"trainer.compute_budget_flops={target_flops}",
        f"init_checkpoint={selected_sft.relative_to('/workspace')}",
    )
    _require_checkpoint(final_control, "Compute-matched continued SFT")
    _run(
        config,
        "evaluate",
        "experiment.phase=continued_sft_evaluation",
        f"checkpoint={final_control.relative_to('/workspace')}",
    )
    _run(config, "report_exp16", "experiment.phase=report")


if __name__ == "__main__":
    main()
