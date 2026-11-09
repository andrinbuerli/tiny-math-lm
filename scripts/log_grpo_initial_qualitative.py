"""Log an initial-policy qualitative sample for a fresh GRPO run."""

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
from tiny_math_lm.data.char_tokenizer import CharacterTokenizer
from tiny_math_lm.data.load import SELECTED_MODULES
from tiny_math_lm.eval.qualitative import build_qualitative_table
from tiny_math_lm.generation import RolloutConfig
from tiny_math_lm.model.gpt import GPTConfig, TinyMathGPT
from tiny_math_lm.tracking.provenance import read_wandb_run_id
from tiny_math_lm.tracking.wandb import finish_wandb, initialize_tracked_run
from tiny_math_lm.train.runner import read_jsonl

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if set(config.data.modules) != set(SELECTED_MODULES):
        raise ValueError("Qualitative sampling is restricted to the five selected modules")
    if not config.init_checkpoint:
        raise ValueError("Set init_checkpoint to the selected raw SFT checkpoint")

    bundle_dir = workspace_path(str(config.data_bundle))
    checkpoint_path = workspace_path(str(config.init_checkpoint))
    if not (bundle_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Prepared data manifest not found: {bundle_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SFT checkpoint not found: {checkpoint_path}")

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_hash = sha256_file(bundle_dir / "manifest.json")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hash = sha256_file(checkpoint_path)
    if payload.get("tokenizer_sha256") != manifest["tokenizer"]["sha256"]:
        raise ValueError("SFT checkpoint tokenizer does not match the selected bundle")
    checkpoint_step = int(payload.get("step", payload.get("update", 0)))

    model_name = str(config.model.name)
    seed = int(config.seed)
    training_run_dir = workspace_path(
        "runs", str(config.experiment.name), model_name, "grpo", f"seed{seed}"
    )
    training_run_id = read_wandb_run_id(training_run_dir)
    run_dir = workspace_path(
        "runs", str(config.experiment.name), model_name,
        "qualitative", "grpo-initial", f"seed{seed}",
    )
    run = initialize_tracked_run(
        config,
        run_dir,
        job_type="qualitative",
        job_metadata={
            "phase": "qualitative",
            "model": model_name,
            "arm": "grpo",
            "source_arm": "sft",
            "checkpoint_selector": "best",
            "checkpoint_step": checkpoint_step,
            "checkpoint_hash": checkpoint_hash,
        },
    )
    exit_code = 0
    try:
        tokenizer = CharacterTokenizer.load(bundle_dir / "char_tokenizer.json")
        model_config = GPTConfig.from_mapping(payload["model_config"])
        # Keep this short initial probe off the GPU while training workers are active.
        model = TinyMathGPT(model_config).cpu()
        model.load_state_dict(payload["model"])
        table, count, accuracy = build_qualitative_table(
            model=model,
            tokenizer=tokenizer,
            split_rows={
                "train": read_jsonl(bundle_dir / "rlvr.jsonl"),
                "validation": read_jsonl(bundle_dir / "sft_validation.jsonl"),
                "test": read_jsonl(bundle_dir / "test.jsonl"),
            },
            training_arm="grpo",
            seed=seed,
            max_new_tokens=int(config.eval.max_new_tokens),
            examples_per_module=int(config.qualitative.examples_per_module),
            rollout=RolloutConfig.from_config(config.get("rollout")),
        )
        source_sft_run_id = read_wandb_run_id(checkpoint_path.parent)
        run.config.update(
            {
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "bundle_manifest_sha256": manifest_hash,
                "tokenizer_sha256": tokenizer.sha256,
                "training_arm": "grpo",
                "qualitative_policy_stage": "initial_sft_policy_before_grpo_update_1",
                "source_grpo_wandb_run_id": training_run_id,
                "source_sft_checkpoint": str(checkpoint_path),
                "source_sft_checkpoint_sha256": checkpoint_hash,
                "source_sft_checkpoint_step": checkpoint_step,
                "source_sft_wandb_run_id": source_sft_run_id,
            },
            allow_val_change=True,
        )
        run.log(
            {
                "optimizer_step": 0,
                "qualitative/examples": table,
                "qualitative/example_count": count,
                **{
                    f"{split}/greedy_accuracy": value
                    for split, value in accuracy.items()
                },
            }
        )
        run.summary.update(
            {
                "qualitative/example_count": count,
                "qualitative/source_grpo_wandb_run_id": training_run_id,
                "qualitative/source_sft_checkpoint_sha256": checkpoint_hash,
                "qualitative/optimizer_step": 0,
            }
        )
        LOGGER.info(
            "Logged %d initial-policy qualitative examples for GRPO run %s",
            count,
            training_run_id,
        )
    except BaseException:
        exit_code = 1
        raise
    finally:
        finish_wandb(run, exit_code=exit_code)


if __name__ == "__main__":
    main()
