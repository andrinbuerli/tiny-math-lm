"""Hydra entry point for fixed-split greedy checkpoint evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.data.load import SELECTED_MODULES
from tinymath.eval.evaluate import evaluate_checkpoint
from tinymath.repro import seed_everything
from tinymath.tracking.wandb import finish_wandb, initialize_wandb

LOGGER = logging.getLogger(__name__)


def _bundle_dir(config: DictConfig) -> Path:
    if config.data_bundle:
        bundle = workspace_path(str(config.data_bundle))
        if not (bundle / "manifest.json").is_file():
            raise FileNotFoundError(f"Prepared data manifest not found: {bundle}")
        return bundle
    candidates = sorted(
        workspace_path("data", "processed").glob(
            f"*-{config.experiment.name}-seed{int(config.seed)}"
        )
    )
    if len(candidates) != 1 or not (candidates[0] / "manifest.json").is_file():
        raise FileNotFoundError("Could not identify one prepared bundle; set data_bundle=... explicitly")
    return candidates[0]


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if set(config.data.modules) != set(SELECTED_MODULES):
        raise ValueError(f"The experiment is restricted to exactly these modules: {SELECTED_MODULES}")
    if not config.checkpoint:
        raise ValueError("Set checkpoint= to the model checkpoint to evaluate")
    seed_everything(int(config.seed), cpu_threads=int(config.runtime.cpu_threads))
    checkpoint = workspace_path(str(config.checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
    bundle_dir = _bundle_dir(config)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    run_dir = workspace_path(
        "runs", str(config.experiment.name), str(config.model.name), "evaluation",
        checkpoint.parent.parent.name, checkpoint.stem, f"seed{int(config.seed)}",
    )
    run = initialize_wandb(config, run_dir, job_type="evaluation")
    try:
        run.config.update(
            {
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "evaluated_checkpoint": str(checkpoint),
                "evaluation_splits": list(config.eval.splits),
            },
            allow_val_change=True,
        )
        summary = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            bundle_dir=bundle_dir,
            output_dir=run_dir,
            split_names=list(config.eval.splits),
            max_new_tokens=int(config.eval.max_new_tokens),
            greedy=bool(config.eval.greedy),
            temperature=float(config.eval.temperature),
            batch_size=int(config.eval.batch_size),
            max_examples_per_split=(
                int(config.eval.max_examples_per_split)
                if config.eval.max_examples_per_split is not None else None
            ),
        )
        run.config.update(
            {
                "training_arm": summary["training_arm"],
                "source_sft_checkpoint": summary["source_sft_checkpoint"],
                "source_training_run_dir": str(checkpoint.parent),
                "checkpoint_step": summary["checkpoint_step"],
                "model_parameters": summary["model_parameters"],
            },
            allow_val_change=True,
        )
        logged = {}
        for split, result in summary["splits"].items():
            logged[f"eval/{split}/examples"] = result["examples"]
            for metric in ("accuracy", "format_validity", "mean_output_tokens"):
                value = result[metric]
                if value is not None:
                    logged[f"eval/{split}/{metric}"] = value
            for module, metrics in result["module"].items():
                logged[f"eval/{split}/module/{module}/accuracy"] = metrics["accuracy"]
                logged[f"eval/{split}/module/{module}/format_validity"] = metrics["format_validity"]
        run.log(logged)
        run.summary.update(
            {
                "evaluation/checkpoint": str(checkpoint),
                "evaluation/results_path": str(run_dir / "results.json"),
                "evaluation/split_manifest_sha256": manifest["split_manifest_sha256"],
            }
        )
        LOGGER.info("Evaluation finished: %s", summary["splits"])
    finally:
        finish_wandb(run)


if __name__ == "__main__":
    main()
