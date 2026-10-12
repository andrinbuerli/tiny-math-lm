"""Hydra entry point for GRPO optimization from an SFT checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.data.load import SELECTED_MODULES
from tinymath.repro import seed_everything
from tinymath.tracking.wandb import finish_wandb, initialize_wandb
from tinymath.train.grpo_runner import train_grpo

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
    if not config.init_checkpoint:
        raise ValueError("Set init_checkpoint to the corresponding SFT checkpoint")
    seed_everything(int(config.seed), cpu_threads=int(config.runtime.cpu_threads))
    init_checkpoint = workspace_path(str(config.init_checkpoint))
    if not init_checkpoint.is_file():
        raise FileNotFoundError(f"SFT checkpoint not found: {init_checkpoint}")
    checkpoint = torch_load_checkpoint(init_checkpoint)
    bundle_dir = _bundle_dir(config, checkpoint.get("dataset_revision"))
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    run_dir = workspace_path(
        "runs", str(config.experiment.name), str(config.model.name), "grpo", f"seed{int(config.seed)}"
    )
    run = initialize_wandb(config, run_dir, job_type="grpo")
    try:
        run.config.update(
            {
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "training_arm": "grpo",
                "source_sft_checkpoint": str(init_checkpoint),
                "source_sft_step": checkpoint["step"],
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
        finish_wandb(run)


def torch_load_checkpoint(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
