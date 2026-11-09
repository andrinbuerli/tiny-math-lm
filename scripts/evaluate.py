"""Hydra entry point for fixed-split greedy checkpoint evaluation."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from tiny_math_lm.checkpoint import sha256_file
from tiny_math_lm.config import workspace_path
from tiny_math_lm.data.load import SELECTED_MODULES
from tiny_math_lm.eval.evaluate import evaluate_checkpoint
from tiny_math_lm.eval.hf_evaluate import evaluate_hf_checkpoint
from tiny_math_lm.generation import RolloutConfig
from tiny_math_lm.runtime_gpu import wait_for_gpu_capacity
from tiny_math_lm.tracking.provenance import read_wandb_run_id
from tiny_math_lm.tracking.wandb import finish_wandb, initialize_tracked_run

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
    wait_for_gpu_capacity(config, str(config.model.name))
    if str(config.model.get("backend", "tinymath")) == "huggingface":
        checkpoint = workspace_path(str(config.checkpoint)) if config.checkpoint else None
        if checkpoint is not None and not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"Hugging Face model checkpoint directory not found: {checkpoint}")
        bundle_dir = _bundle_dir(config)
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        checkpoint_identity_path = checkpoint / "experiment_identity.json" if checkpoint else None
        checkpoint_identity = (
            json.loads(checkpoint_identity_path.read_text(encoding="utf-8"))
            if checkpoint_identity_path and checkpoint_identity_path.is_file() else {}
        )
        arm = (
            "base_zero_shot" if checkpoint is None
            else str(checkpoint_identity.get("training_arm", checkpoint.parent.parent.name))
        )
        identity = "pristine" if checkpoint is None else checkpoint.name
        run_dir = workspace_path(
            "runs", str(config.experiment.name), str(config.model.name), "evaluation",
            arm, identity, f"seed{int(config.seed)}",
        )
        checkpoint_sha256 = checkpoint_identity.get("checkpoint_sha256")
        run = initialize_tracked_run(
            config,
            run_dir,
            job_type="checkpoint-evaluation",
            job_metadata={
                "phase": str(config.experiment.phase),
                "model": str(config.model.name),
                "arm": arm,
                "checkpoint": str(checkpoint) if checkpoint else "pinned_pristine_base",
                "checkpoint_sha256": checkpoint_sha256,
                "formats": ["ordinary_qa", "reasoning_executor"],
            },
        )
        try:
            run.config.update(
                {
                    "dataset_revision": manifest["dataset_revision"],
                    "split_manifest_sha256": manifest["split_manifest_sha256"],
                    "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
                    "evaluated_checkpoint": str(checkpoint) if checkpoint else "pinned_pristine_base",
                    "evaluated_checkpoint_sha256": checkpoint_sha256,
                    "evaluation_splits": list(config.eval.splits),
                    "source_training_wandb_run_id": (
                        read_wandb_run_id(checkpoint.parent) if checkpoint else None
                    ),
                },
                allow_val_change=True,
            )
            summary = evaluate_hf_checkpoint(
                config=config,
                bundle_dir=bundle_dir,
                output_dir=run_dir,
                checkpoint=checkpoint,
                split_names=list(config.eval.splits),
                max_examples_per_split=(
                    int(config.eval.max_examples_per_split)
                    if config.eval.max_examples_per_split is not None else None
                ),
                run=run,
            )
            run.summary.update(
                {
                    "evaluation/results_path": str(run_dir / "results.json"),
                    "evaluation/checkpoint": str(checkpoint) if checkpoint else "pinned_pristine_base",
                    "evaluation/checkpoint_sha256": checkpoint_sha256,
                    "evaluation/split_manifest_sha256": manifest["split_manifest_sha256"],
                }
            )
            LOGGER.info("Hugging Face evaluation finished: %s", summary["splits"])
        finally:
            finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)
        return
    if not config.checkpoint:
        raise ValueError("Set checkpoint= to the model checkpoint to evaluate")
    checkpoint = workspace_path(str(config.checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint}")
    bundle_dir = _bundle_dir(config)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(checkpoint_payload.get("step", checkpoint_payload.get("update", 0)))
    checkpoint_arm = str(
        checkpoint_payload.get("training_arm")
        or checkpoint_payload.get("stage")
        or "unknown-arm"
    )
    checkpoint_selector = checkpoint.stem
    checkpoint_hash = sha256_file(checkpoint)
    if int(config.experiment.get("number", 0) or 0) == 11:
        evaluation_identity = f"step{checkpoint_step}-{checkpoint_hash[:12]}"
        run_dir = workspace_path(
            "runs",
            str(config.experiment.name),
            str(config.model.name),
            "evaluation",
            checkpoint_arm,
            checkpoint_selector,
            evaluation_identity,
            f"seed{int(config.seed)}",
        )
    else:
        run_dir = workspace_path(
            "runs", str(config.experiment.name), str(config.model.name), "evaluation",
            checkpoint.parent.parent.name, checkpoint.stem, f"seed{int(config.seed)}",
        )
    run = initialize_tracked_run(
        config,
        run_dir,
        job_type="evaluation",
        job_metadata={
            "phase": "evaluation",
            "model": str(config.model.name),
            "arm": checkpoint_arm,
            "source_arm": checkpoint_arm,
            "checkpoint_selector": checkpoint_selector,
            "checkpoint_step": checkpoint_step,
            "checkpoint_hash": checkpoint_hash,
        },
    )
    try:
        run.config.update(
            {
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "evaluated_checkpoint": str(checkpoint),
                "evaluated_checkpoint_sha256": checkpoint_hash,
                "protocol": manifest.get("annotation", {}).get("protocol", "answer_only"),
                "annotation_sha256": manifest.get("annotation", {}).get("annotation_sha256"),
                "bundle_sha256": sha256_file(bundle_dir / "manifest.json"),
                "evaluation_splits": list(config.eval.splits),
                "source_training_run_dir": str(checkpoint.parent),
                "source_training_wandb_run_id": read_wandb_run_id(checkpoint.parent),
            },
            allow_val_change=True,
        )
        summary = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            bundle_dir=bundle_dir,
            output_dir=run_dir,
            split_names=list(config.eval.splits),
            max_new_tokens=int(config.eval.max_new_tokens),
            rollout=RolloutConfig.from_config(config.get("rollout")),
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
                "rollout": summary.get("rollout"),
                "assisted": summary.get("assisted", False),
                "training_arm": summary["training_arm"],
                "source_sft_checkpoint": summary["source_sft_checkpoint"],
                "source_sft_wandb_run_id": (
                    read_wandb_run_id(Path(summary["source_sft_checkpoint"]).parent)
                    if summary["source_sft_checkpoint"]
                    else None
                ),
                "checkpoint_step": summary["checkpoint_step"],
                "model_parameters": summary["model_parameters"],
                "protocol": summary["protocol"],
                "annotation_sha256": summary["annotation_sha256"],
                "bundle_sha256": summary["bundle_sha256"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
            },
            allow_val_change=True,
        )
        logged = {}
        for split, result in summary["splits"].items():
            logged[f"eval/{split}/examples"] = result["examples"]
            for metric in (
                "accuracy",
                "format_validity",
                "trace_structure_validity",
                "trace_reference_exact_match_rate",
                "mean_output_tokens",
                "mean_reward", "full_success_rate", "format_only_rate", "mean_format_reward",
                "mean_success_reward", "call_correctness", "calls_per_completion",
                "no_call_success_rate", "executor_failures", "policy_tokens", "executor_tokens",
                "call_policy_violation_rate",
            ):
                value = result[metric]
                if value is not None:
                    logged[f"eval/{split}/{metric}"] = value
            for module, metrics in result["module"].items():
                for metric in (
                    "accuracy",
                    "format_validity",
                    "trace_structure_validity",
                    "trace_reference_exact_match_rate",
                ):
                    value = metrics[metric]
                    if value is not None:
                        logged[f"eval/{split}/module/{module}/{metric}"] = value
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
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
