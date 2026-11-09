"""Hydra entry point for the fixed five-module data preparation protocol."""

from __future__ import annotations

import logging
import sys
import time

import hydra
from omegaconf import DictConfig

from tiny_math_lm.config import read_saved_resolved_config, workspace_path
from tiny_math_lm.data.load import SELECTED_MODULES, inspect_dataset, stream_english_rows
from tiny_math_lm.data.prepare import prepare_bundle
from tiny_math_lm.repro import seed_everything
from tiny_math_lm.tracking.wandb import finish_wandb, initialize_wandb

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if set(config.data.modules) != set(SELECTED_MODULES):
        raise ValueError(f"The experiment is restricted to exactly these modules: {SELECTED_MODULES}")
    seed_everything(int(config.seed), cpu_threads=int(config.runtime.cpu_threads))
    run_dir = workspace_path(
        "runs", str(config.experiment.name), "data", f"seed{int(config.seed)}"
    )
    previous_config = read_saved_resolved_config(run_dir)
    revision = config.data.revision
    if revision is None and previous_config:
        revision = previous_config.get("data", {}).get("revision")
    inspection = inspect_dataset(config.data.dataset, revision)
    config.data.revision = inspection.revision
    run = initialize_wandb(config, run_dir, job_type="data-preparation")
    try:
        run.define_metric("data/selected_stream_rows")
        run.define_metric("data/*", step_metric="data/selected_stream_rows")
        LOGGER.info(
            "Inspected %s config=%s split=%s rows=%d revision=%s",
            inspection.dataset_id,
            inspection.config,
            inspection.split,
            inspection.split_examples,
            inspection.revision,
        )
        run.config.update(
            {
                "dataset_revision": inspection.revision,
                "dataset_schema": inspection.features,
                "dataset_modules": list(SELECTED_MODULES),
                "dataset_split": inspection.split,
            },
            allow_val_change=True,
        )
        start = time.monotonic()
        records = []
        for index, record in enumerate(
            stream_english_rows(
                dataset_id=config.data.dataset,
                revision=inspection.revision,
                modules=tuple(config.data.modules),
            ),
            start=1,
        ):
            records.append(record)
            if index % 250_000 == 0:
                LOGGER.info("Read %d rows from the five selected modules", index)
                run.log(
                    {
                        "data/selected_stream_rows": index,
                        "data/scan_seconds": time.monotonic() - start,
                    }
                )
        scan_seconds = time.monotonic() - start
        output_dir = workspace_path(
            "data",
            "processed",
            f"{inspection.revision[:12]}-{config.experiment.name}-seed{int(config.seed)}",
        )
        pools, tokenizer, manifest = prepare_bundle(
            records,
            inspection,
            output_dir=output_dir,
            seed=int(config.seed),
            sft_examples=int(config.data.sft_examples),
            validation_fraction=float(config.data.validation_fraction),
            rlvr_fraction=float(config.data.rlvr_fraction),
            context_length=int(config.data.context_length),
            ood_examples_per_module=int(config.data.ood_examples_per_module),
        )
        run.config.update(
            {
                "split_manifest_sha256": manifest["split_manifest_sha256"],
                "tokenizer_sha256": tokenizer.sha256,
                "split_counts": {name: len(rows) for name, rows in pools.splits.items()},
                "data_output_dir": str(output_dir),
            },
            allow_val_change=True,
        )
        metrics = {
            "data/selected_stream_rows": len(records),
            "data/scan_seconds": scan_seconds,
            "data/unique_questions": manifest["unique_selected_source_questions"],
            "data/duplicate_questions_removed": pools.duplicate_questions,
            "data/tokenizer_vocab_size": tokenizer.vocab_size,
        }
        metrics.update(
            {f"data/split/{name}_examples": len(rows) for name, rows in pools.splits.items()}
        )
        run.log(metrics)
        LOGGER.info("Prepared data bundle at %s", output_dir)
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
