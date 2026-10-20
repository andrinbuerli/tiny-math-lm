"""Hydra entry point for GRPO optimization from an SFT checkpoint."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.data.load import SELECTED_MODULES
from tinymath.runtime_gpu import wait_for_gpu_capacity
from tinymath.tracking.provenance import read_wandb_run_id
from tinymath.tracking.wandb import finish_wandb, initialize_tracked_run
from tinymath.train.budgets import isolated_training_run_dir
from tinymath.train.grpo_runner import train_grpo
from tinymath.train.hf_grpo_runner import train_hf_grpo

LOGGER = logging.getLogger(__name__)


def _bundle_dir(config: DictConfig, revision: str | None = None) -> Path:
    if config.data_bundle:
        bundle = workspace_path(str(config.data_bundle))
        if not (bundle / "manifest.json").is_file():
            raise FileNotFoundError(f"Prepared data manifest not found: {bundle}")
        return bundle
    pattern = f"*-{config.experiment.name}-seed{int(config.seed)}"
    candidates = sorted(workspace_path("data", "processed").glob(pattern))
    if revision:
        candidates = [path for path in candidates if path.name.startswith(revision[:12])]
    if len(candidates) != 1 or not (candidates[0] / "manifest.json").is_file():
        raise FileNotFoundError("Could not identify one prepared bundle; set data_bundle=... explicitly")
    return candidates[0]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if set(config.data.modules) != set(SELECTED_MODULES):
        raise ValueError(f"The experiment is restricted to exactly these modules: {SELECTED_MODULES}")
    if str(config.model.get("backend", "tinymath")) == "huggingface":
        init_checkpoint = workspace_path(str(config.init_checkpoint)) if config.init_checkpoint else None
        if init_checkpoint is not None and not init_checkpoint.is_dir():
            raise FileNotFoundError(f"Hugging Face SFT checkpoint directory not found: {init_checkpoint}")
        if init_checkpoint is None and str(config.experiment.phase) not in {"pilot", "smoke"}:
            raise ValueError("Main Experiment 16 GRPO requires the selected SFT checkpoint")
        bundle_dir = _bundle_dir(config)
        wait_for_gpu_capacity(config, str(config.model.name))
        run_dir = workspace_path(
            "runs", str(config.experiment.name), str(config.model.name), "grpo",
            str(config.experiment.phase), f"seed{int(config.seed)}",
        )
        run = initialize_tracked_run(
            config,
            run_dir,
            job_type="grpo",
            job_metadata={"phase": str(config.experiment.phase), "training_arm": "grpo"},
        )
        try:
            run.config.update(
                {
                    "source_sft_wandb_run_id": (
                        read_wandb_run_id(init_checkpoint.parent)
                        if init_checkpoint is not None else None
                    ),
                },
                allow_val_change=True,
            )
            result = train_hf_grpo(
                config=config,
                bundle_dir=bundle_dir,
                init_checkpoint=init_checkpoint,
                run_dir=run_dir,
                run=run,
            )
            run.summary.update(
                {
                    "grpo/updates": result["update"],
                    "grpo/prompt_examples": result["prompt_examples"],
                    "grpo/tokens_seen": result["tokens_seen"],
                    "grpo/policy_tokens": result["policy_tokens"],
                    "grpo/executor_tokens": result["executor_tokens"],
                    "grpo/estimated_flops": result["estimated_flops"],
                    "grpo/final_reward": result["final_reward"],
                    "grpo/final_loss": result["final_loss"],
                    "grpo/wall_seconds": result["wall_seconds"],
                    "grpo/tokens_per_second": result["tokens_per_second"],
                    "grpo/peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                    "grpo/checkpoint": result["checkpoint"],
                    "grpo/checkpoint_sha256": result["checkpoint_sha256"],
                    "grpo/latest_checkpoint": result["latest_checkpoint"],
                }
            )
            LOGGER.info("Hugging Face GRPO finished: %s", result)
        finally:
            finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)
        return
    if not config.init_checkpoint:
        raise ValueError("Set init_checkpoint to the corresponding SFT checkpoint")
    init_checkpoint = workspace_path(str(config.init_checkpoint))
    if not init_checkpoint.is_file():
        raise FileNotFoundError(f"SFT checkpoint not found: {init_checkpoint}")
    checkpoint = torch_load_checkpoint(init_checkpoint)
    bundle_dir = _bundle_dir(config, checkpoint.get("dataset_revision"))
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    run_dir = Path(
        isolated_training_run_dir(
            config,
            model=str(config.model.name),
            arm="grpo",
            runs_root=str(workspace_path("runs")),
        )
    )
    run = initialize_tracked_run(config, run_dir, job_type="grpo")
    try:
        run.config.update(
            {
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "training_arm": "grpo",
                "source_sft_checkpoint": str(init_checkpoint),
                "source_sft_step": checkpoint["step"],
                "source_sft_wandb_run_id": read_wandb_run_id(init_checkpoint.parent),
            },
            allow_val_change=True,
        )
        result = train_grpo(
            config=config,
            bundle_dir=bundle_dir,
            init_checkpoint=init_checkpoint,
            run_dir=run_dir,
            run=run,
        )
        run.summary.update(
            {
                "grpo/updates": result["update"],
                "grpo/prompt_examples": result["prompt_examples"],
                "grpo/examples_drawn": result["examples_drawn"],
                "grpo/nonpadding_input_tokens": result["nonpadding_input_tokens"],
                "grpo/generated_tokens": result["generated_tokens"],
                "grpo/padded_compute_positions": result["padded_compute_positions"],
                "grpo/tokens_seen": result["tokens_seen"],
                "grpo/estimated_flops": result["estimated_flops"],
                "grpo/final_reward": result["final_reward"],
                "grpo/checkpoint": result["checkpoint"],
                "grpo/latest_inference_checkpoint": result["inference_checkpoint"],
                "grpo/source_sft_checkpoint": str(init_checkpoint),
            }
        )
        LOGGER.info("GRPO training finished: %s", result)
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


def torch_load_checkpoint(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
