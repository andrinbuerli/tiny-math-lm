"""Hydra comparison report entry point with W&B provenance."""

from __future__ import annotations

import logging

import hydra
import wandb
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.reporting import make_report
from tinymath.tracking.wandb import finish_wandb, initialize_wandb

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    run_dir = workspace_path(
        "runs", str(config.experiment.name), "comparison", f"seed{int(config.seed)}"
    )
    output_dir = workspace_path(str(config.reporting.output_dir))
    run = initialize_wandb(config, run_dir, job_type="comparison")
    try:
        report, outputs = make_report(
            runs_dir=workspace_path("runs"),
            experiment=str(config.experiment.name),
            output_dir=output_dir,
        )
        run.log({"comparison/available_evaluations": report["coverage"]["available_evaluations"]})
        for artifact in outputs:
            if artifact.suffix == ".png":
                run.log({f"comparison/{artifact.stem}": wandb.Image(str(artifact))})
        run.summary.update(
            {
                "comparison/report_path": str(output_dir / "comparison.json"),
                "comparison/csv_path": str(output_dir / "comparison.csv"),
                "comparison/available_evaluations": report["coverage"]["available_evaluations"],
            }
        )
        LOGGER.info("Comparison report written to %s", output_dir)
    finally:
        finish_wandb(run)


if __name__ == "__main__":
    main()
