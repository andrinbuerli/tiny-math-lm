"""Resume a training W&B run and attach deterministic example predictions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.eval.qualitative import log_qualitative_examples
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.tracking.wandb import finish_wandb, initialize_wandb
from tinymath.train.runner import read_jsonl

LOGGER = logging.getLogger(__name__)


def _bundle_dir(config: DictConfig) -> tuple[Path, dict[str, Any]]:
    if config.get("data_bundle"):
        bundle = workspace_path(str(config.data_bundle))
    else:
        matches = sorted(
            workspace_path("data", "processed").glob(
                f"*-{config.experiment.name}-seed{int(config.seed)}"
            )
        )
        if len(matches) != 1:
            raise FileNotFoundError("Set data_bundle=... to select one prepared data pool")
        bundle = matches[0]
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared data manifest not found: {manifest_path}")
    return bundle, json.loads(manifest_path.read_text(encoding="utf-8"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    bundle_dir, manifest = _bundle_dir(config)
    if config.trainer.name == "continued_sft":
        stage = "continued_sft"
        training_split = "sft_train"
        checkpoint_path = workspace_path(
            "runs", config.experiment.name, config.model.name, stage,
            f"seed{int(config.seed)}", "best.pt",
        )
    elif config.init_checkpoint:
        stage = "grpo"
        training_split = "rlvr"
        checkpoint_path = workspace_path(
            "runs", config.experiment.name, config.model.name, stage,
            f"seed{int(config.seed)}", "latest.pt",
        )
    else:
        stage = "sft"
        training_split = "sft_train"
        checkpoint_path = workspace_path(
            "runs", config.experiment.name, config.model.name, stage,
            f"seed{int(config.seed)}", "best.pt",
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_path.parent
    run = initialize_wandb(config, run_dir, job_type=stage)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        tokenizer = CharacterTokenizer.load(bundle_dir / "char_tokenizer.json")
        if payload.get("tokenizer_sha256") != tokenizer.sha256:
            raise ValueError("Training checkpoint tokenizer does not match the selected data bundle")
        model_config = GPTConfig.from_mapping(payload["model_config"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyMathGPT(model_config).to(device)
        model.load_state_dict(payload["model"])
        count = log_qualitative_examples(
            model=model,
            tokenizer=tokenizer,
            split_rows={
                "train": read_jsonl(bundle_dir / f"{training_split}.jsonl"),
                "validation": read_jsonl(bundle_dir / "sft_validation.jsonl"),
                "test": read_jsonl(bundle_dir / "test.jsonl"),
            },
            run=run,
            training_arm=stage,
            seed=int(config.seed),
            step=int(payload.get("step", payload.get("update", 0))),
            max_new_tokens=int(config.eval.max_new_tokens),
        )
        run.config.update(
            {
                "dataset_revision": manifest["dataset_revision"],
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": tokenizer.sha256,
                "qualitative_checkpoint": str(checkpoint_path),
            },
            allow_val_change=True,
        )
        LOGGER.info("Logged %d qualitative examples to the %s run", count, stage)
    finally:
        finish_wandb(run)


if __name__ == "__main__":
    main()
