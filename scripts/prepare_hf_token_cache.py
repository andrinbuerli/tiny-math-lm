"""Hydra entry point for Experiment 16 native-token cache preparation."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from pathlib import Path

import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from tiny_math_lm.config import workspace_path
from tiny_math_lm.data.hf_reasoning import sha256_file, validate_source_bundle
from tiny_math_lm.data.hf_token_cache import prepare_hf_token_cache
from tiny_math_lm.tracking.wandb import finish_wandb, initialize_tracked_run

LOGGER = logging.getLogger(__name__)


def _bundle_dir(config: DictConfig) -> Path:
    path = workspace_path(str(config.data_bundle))
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"Prepared reasoning-v3 bundle not found: {path}")
    return path


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if int(config.experiment.get("number", 0) or 0) != 16:
        raise ValueError("prepare_hf_token_cache.py is reserved for named Experiment 16 runs")
    bundle_dir = _bundle_dir(config)
    expected_manifest_hash = str(config.checkpoint_provenance.data_bundle_manifest_sha256)
    source_manifest = validate_source_bundle(
        bundle_dir,
        expected_manifest_sha256=expected_manifest_hash,
        expected_seed=int(config.seed),
    )
    run_dir = workspace_path(
        "runs",
        str(config.experiment.name),
        "preparation",
        str(config.experiment.phase),
        f"seed{int(config.seed)}",
    )
    run = initialize_tracked_run(
        config,
        run_dir,
        job_type="native_token_cache",
        job_metadata={
            "phase": "data_preparation",
            "source_bundle_manifest_sha256": expected_manifest_hash,
            "model_revision": str(config.model.revision),
        },
    )
    try:
        cache_dir = workspace_path(str(config.model.cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(
            str(config.model.repo_id),
            revision=str(config.model.revision),
            cache_dir=str(cache_dir),
            trust_remote_code=bool(config.model.trust_remote_code),
            use_fast=True,
        )
        config_path = hf_hub_download(
            repo_id=str(config.model.repo_id),
            filename="config.json",
            revision=str(config.model.revision),
            cache_dir=str(cache_dir),
        )
        tokenizer_cache, cache_manifest = prepare_hf_token_cache(
            bundle_dir=bundle_dir,
            cache_root=workspace_path("data", "cache", "tokenized"),
            tokenizer=tokenizer,
            tokenizer_revision=str(config.model.revision),
            config_sha256=sha256_file(config_path),
            expected_manifest_sha256=expected_manifest_hash,
            expected_eval_identities=OmegaConf.to_container(
                config.checkpoint_provenance.fixed_evaluation_question_identity_sha256,
                resolve=True,
            ),
            context_length=int(config.model.context_length),
            completion_cap=int(config.rlvr.max_completion_tokens),
        )
        packages = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "huggingface-hub")
        }
        run.config.update(
            {
                "dataset_revision": source_manifest["dataset_revision"],
                "source_bundle_manifest_sha256": expected_manifest_hash,
                "source_split_manifest_sha256": source_manifest["split_manifest_sha256"],
                "model_repo_id": str(config.model.repo_id),
                "model_revision": str(config.model.revision),
                "model_config_sha256": cache_manifest["model_config_sha256"],
                "tokenizer_sha256": cache_manifest["tokenizer_sha256"],
                "tokenizer_identity": cache_manifest["tokenizer_identity"],
                "formatting_sha256": cache_manifest["formatting_sha256"],
                "package_versions": packages,
                "token_cache_dir": str(tokenizer_cache),
            },
            allow_val_change=True,
        )
        logged = {}
        for split, values in cache_manifest["splits"].items():
            logged[f"data/{split}/rows"] = values["rows"]
            for metric, value in values["token_lengths"].items():
                logged[f"data/{split}/token_length_{metric}"] = value
            logged[f"data/{split}/completion_token_max"] = values["completion_tokens"]["max"]
        run.log(logged)
        run.summary.update(
            {
                "data/token_cache_dir": str(tokenizer_cache),
                "data/training_parent_rows": cache_manifest["sft_training_population"][
                    "parent_rows"
                ],
                "data/training_eligible_subtask_rows": cache_manifest[
                    "sft_training_population"
                ]["eligible_standalone_subtask_rows"],
                "data/training_rows": cache_manifest["sft_training_population"]["total_rows"],
                "data/tokenizer_sha256": cache_manifest["tokenizer_sha256"],
                "data/config_sha256": cache_manifest["model_config_sha256"],
                "data/manifest": str(tokenizer_cache / "manifest.json"),
            }
        )
        LOGGER.info("Native token cache prepared: %s", tokenizer_cache)
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
