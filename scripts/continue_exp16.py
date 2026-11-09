"""Hydra entry point for resuming Experiment 16 after the main SFT run."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from tiny_math_lm.config import workspace_path

LOGGER = logging.getLogger(__name__)
EVAL_BATCH_SIZE = 4


def _run(config: DictConfig, script: str, *overrides: str) -> None:
    uv = shutil.which("uv") or "/workspace/.tools/uv"
    command = [
        uv,
        "run",
        "--locked",
        "python",
        f"scripts/{script}.py",
        f"experiment={config.experiment.config_group}",
        *overrides,
    ]
    LOGGER.info("Running Experiment 16 continuation stage: %s", " ".join(command[3:]))
    subprocess.run(command, cwd="/workspace", check=True)


def _evaluate(config: DictConfig, phase: str, checkpoint: str) -> None:
    _run(
        config,
        "evaluate",
        f"experiment.phase={phase}",
        f"checkpoint={checkpoint}",
        f"eval.batch_size={EVAL_BATCH_SIZE}",
    )


def _require_checkpoint(path: Path, label: str) -> None:
    required = ("config.json", "experiment_identity.json", "model.safetensors")
    if not all((path / name).is_file() for name in required):
        raise FileNotFoundError(f"{label} checkpoint is missing or incomplete: {path}")


def _read_tracking(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "wandb_run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_finished(run_dir: Path, *required_files: str) -> bool:
    tracking = _read_tracking(run_dir)
    return tracking.get("run_status") == "finished" and all(
        (run_dir / relative).is_file() for relative in required_files
    )


def _is_sft_process(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"train_sft.py" in command and b"experiment=16_qwen3_base_reasoning_v3" in command


def _wait_for_sft(run_dir: Path, pid: int, interval: int) -> dict:
    summary_path = run_dir / "training_summary.json"
    while True:
        summary = {}
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        tracking = _read_tracking(run_dir)
        if summary and tracking.get("run_status") == "finished":
            return summary
        if not _is_sft_process(pid):
            if summary:
                raise RuntimeError(
                    "Main SFT process exited before its tracked run was marked finished"
                )
            raise RuntimeError(
                f"Main SFT process {pid} exited without a durable training summary"
            )
        LOGGER.info("Waiting for main SFT process %d to finish", pid)
        time.sleep(max(10, interval))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if str(config.model.get("backend", "tinymath")) != "huggingface":
        raise ValueError("Experiment 16 continuation requires the Hugging Face backend")
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 16 continuation requires the admitted RTX 4090")

    seed = int(config.seed)
    root = workspace_path("runs", str(config.experiment.name), str(config.model.name))
    sft_run = root / "sft" / "main" / f"seed{seed}"
    selected_sft = sft_run / "best"
    wait_pid = config.orchestration.get("wait_for_sft_pid")
    if wait_pid is not None:
        _wait_for_sft(
            sft_run,
            int(wait_pid),
            int(config.reporting.intermediate_refresh_seconds),
        )
    elif not _stage_finished(sft_run, "training_summary.json"):
        raise RuntimeError(
            "Main SFT is not complete; set orchestration.wait_for_sft_pid to its live PID "
            "to wait for it"
        )
    _require_checkpoint(selected_sft, "Selected SFT")

    sft_eval_dir = root / "evaluation" / "sft" / "best" / f"seed{seed}"
    if not _stage_finished(sft_eval_dir, "results.json"):
        _evaluate(config, "sft_evaluation", str(selected_sft.relative_to("/workspace")))

    grpo_run = root / "grpo" / "main" / f"seed{seed}"
    final_grpo = grpo_run / "final"
    if not _stage_finished(grpo_run, "final/config.json", "final/experiment_identity.json"):
        _run(
            config,
            "train_rlvr",
            "experiment.phase=main",
            f"init_checkpoint={selected_sft.relative_to('/workspace')}",
            "+rlvr.rollout_generation_batch_size=64",
            "rlvr.rollout_chunk_size=4",
        )
    _require_checkpoint(final_grpo, "Final GRPO")

    grpo_eval_dir = root / "evaluation" / "grpo" / "final" / f"seed{seed}"
    if not _stage_finished(grpo_eval_dir, "results.json"):
        _evaluate(config, "grpo_evaluation", str(final_grpo.relative_to("/workspace")))

    grpo_state = torch.load(grpo_run / "latest.pt", map_location="cpu", weights_only=False)
    target_flops = int(grpo_state["estimated_flops"])
    if target_flops <= 0:
        raise ValueError("Completed GRPO checkpoint has no positive FLOP estimate")

    continued_run = root / "continued_sft" / "continued_sft" / f"seed{seed}"
    final_control = continued_run / "final"
    if not _stage_finished(continued_run, "final/config.json", "final/experiment_identity.json"):
        _run(
            config,
            "train_sft",
            "experiment.phase=continued_sft",
            "trainer.name=continued_sft",
            f"trainer.compute_budget_flops={target_flops}",
            f"init_checkpoint={selected_sft.relative_to('/workspace')}",
        )
    _require_checkpoint(final_control, "Compute-matched continued-SFT")

    control_eval_dir = root / "evaluation" / "continued_sft" / "final" / f"seed{seed}"
    if not _stage_finished(control_eval_dir, "results.json"):
        _evaluate(
            config,
            "continued_sft_evaluation",
            str(final_control.relative_to("/workspace")),
        )
    _run(config, "report_exp16", "experiment.phase=report")


if __name__ == "__main__":
    main()
