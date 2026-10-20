"""Hydra entry point for supervised and continued-SFT experiments."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.data.load import SELECTED_MODULES
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.runtime_gpu import wait_for_gpu_capacity
from tinymath.tracking.provenance import read_wandb_run_id
from tinymath.tracking.wandb import finish_wandb, initialize_tracked_run
from tinymath.train.budgets import isolated_training_run_dir
from tinymath.train.hf_sft import train_hf_sft
from tinymath.train.runner import train_sft

LOGGER = logging.getLogger(__name__)


def _bundle_dir(config: DictConfig) -> Path:
    explicit = config.get("data_bundle")
    if explicit:
        path = workspace_path(str(explicit))
        if not (path / "manifest.json").exists():
            raise FileNotFoundError(f"Prepared data manifest not found: {path}")
        return path
    if config.data.revision:
        candidate = workspace_path(
            "data", "processed",
            f"{str(config.data.revision)[:12]}-{config.experiment.name}-seed{int(config.seed)}",
        )
        if (candidate / "manifest.json").exists():
            return candidate
    matches = sorted(workspace_path("data", "processed").glob(
        f"*-{config.experiment.name}-seed{int(config.seed)}"
    ))
    if len(matches) != 1 or not (matches[0] / "manifest.json").exists():
        raise FileNotFoundError(
            "Could not identify one prepared data bundle; run prepare_data.py or set data_bundle=..."
        )
    return matches[0]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if set(config.data.modules) != set(SELECTED_MODULES):
        raise ValueError(f"The experiment is restricted to exactly these modules: {SELECTED_MODULES}")
    if str(config.model.get("backend", "tinymath")) == "huggingface":
        wait_for_gpu_capacity(config, str(config.model.name))
    bundle_dir = _bundle_dir(config)
    manifest = __import__("json").loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    if str(config.model.get("backend", "tinymath")) == "huggingface":
        run_dir = workspace_path(
            "runs", str(config.experiment.name), str(config.model.name),
            str(config.trainer.name), str(config.experiment.phase), f"seed{int(config.seed)}",
        )
    else:
        run_dir = Path(
            isolated_training_run_dir(
                config,
                model=str(config.model.name),
                arm=str(config.trainer.name),
                runs_root=str(workspace_path("runs")),
            )
        )
    run = initialize_tracked_run(config, run_dir, job_type=str(config.trainer.name))
    try:
        init_checkpoint = (
            workspace_path(str(config.init_checkpoint)) if config.init_checkpoint else None
        )
        if str(config.model.get("backend", "tinymath")) == "huggingface":
            if init_checkpoint is not None and not init_checkpoint.exists():
                raise FileNotFoundError(f"Hugging Face initialization checkpoint not found: {init_checkpoint}")
            run.config.update(
                {
                    "training_arm": str(config.trainer.name),
                    "source_sft_checkpoint": (
                        str(init_checkpoint)
                        if str(config.trainer.name) == "continued_sft" and init_checkpoint else None
                    ),
                    "source_sft_wandb_run_id": (
                        read_wandb_run_id(init_checkpoint.parent)
                        if str(config.trainer.name) == "continued_sft" and init_checkpoint else None
                    ),
                },
                allow_val_change=True,
            )
            result = train_hf_sft(
                config=config,
                run_dir=run_dir,
                run=run,
                init_checkpoint=init_checkpoint,
            )
            run.summary.update(
                {
                    "training/optimizer_steps": result["step"],
                    "training/epochs_completed": result["epoch"],
                    "training/examples_used": result["training_examples"],
                    "training/examples_seen": result["examples_seen"],
                    "training/tokens_seen": result["tokens_seen"],
                    "training/supervised_tokens_seen": result["supervised_tokens_seen"],
                    "training/tokens_per_second": result["tokens_per_second"],
                    "training/wall_seconds": result["wall_seconds"],
                    "training/peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                    "training/additional_estimated_flops": result[
                        "additional_estimated_flops"
                    ],
                    "training/best_validation_loss": result["best_validation_loss"],
                    "training/model_parameters": result["parameters"],
                    "training/checkpoint": result["checkpoint"],
                    "training/checkpoint_sha256": result["checkpoint_sha256"],
                    "training/best_checkpoint": result["best_checkpoint"],
                    "training/final_checkpoint": result["final_checkpoint"],
                    "training/compute_budget_flops": result["compute_budget_flops"],
                    "training/compute_overshoot_flops": result["compute_overshoot_flops"],
                    "training/compute_matching_error": result["compute_matching_error"],
                    "training/planned_optimizer_steps": result["planned_optimizer_steps"],
                    "training/planned_epochs": result["planned_epochs"],
                    "training/latest_checkpoint": result["latest_checkpoint"],
                }
            )
            LOGGER.info("Hugging Face SFT finished: %s", result)
            return
        model_values = dict(config.model)
        model_values["vocab_size"] = int(manifest["tokenizer"]["vocab_size"])
        parameter_model = TinyMathGPT(GPTConfig.from_mapping(model_values))
        lock_digest = hashlib.sha256(workspace_path("uv.lock").read_bytes()).hexdigest()
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd="/workspace", text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_sha = "unknown"
        run.config.update(
            {
                "git_sha": git_sha,
                "uv_lock_sha256": lock_digest,
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "model_parameters": parameter_model.count_parameters(),
                "training_arm": "continued_sft" if config.trainer.name == "continued_sft" else "sft",
                "data_bundle": str(bundle_dir),
                "source_sft_wandb_run_id": (
                    read_wandb_run_id(init_checkpoint.parent)
                    if config.trainer.name == "continued_sft" and init_checkpoint is not None
                    else None
                ),
            },
            allow_val_change=True,
        )
        result = train_sft(
            config=config,
            bundle_dir=bundle_dir,
            run_dir=run_dir,
            run=run,
            init_checkpoint=init_checkpoint,
        )
        run.summary.update(
            {
                "training/optimizer_steps": result["step"],
                "training/examples_used": result["training_examples"],
                "training/examples_drawn": result["examples_drawn"],
                "training/nonpadding_input_tokens": result["nonpadding_input_tokens"],
                "training/padded_compute_positions": result["padded_compute_positions"],
                "training/supervised_tokens": result["supervised_tokens"],
                "training/tokens_seen": result["tokens_seen"],
                "training/additional_estimated_flops": result["additional_estimated_flops"],
                "training/best_validation_loss": result["best_validation_loss"],
                "training/model_parameters": result["parameters"],
                "training/checkpoint": result["checkpoint"],
                "training/latest_inference_checkpoint": result["inference_checkpoint"],
                "training/source_sft_checkpoint": result["source_sft_checkpoint"],
            }
        )
        LOGGER.info("SFT training finished: %s", result)
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
