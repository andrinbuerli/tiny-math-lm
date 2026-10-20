"""Hydra entry point for the tracked Experiment 16 comparison and report."""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.eval.exp16_report import build_exp16_report
from tinymath.tracking.wandb import finish_wandb, initialize_tracked_run

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    output_dir = workspace_path(str(config.reporting.output_dir))
    run_dir = output_dir / "comparison"
    run = initialize_tracked_run(
        config,
        run_dir,
        job_type="experimental-comparison",
        job_metadata={
            "phase": str(config.experiment.phase),
            "model": str(config.model.name),
            "arm": "comparison",
            "seed": int(config.seed),
        },
    )
    try:
        report = build_exp16_report(
            runs_root=workspace_path("runs"),
            output_dir=output_dir,
            seed=int(config.seed),
        )
        scalar_logs = {}
        for arm, result in report["results"].items():
            for key, summary in result.get("splits", {}).items():
                for metric in (
                    "accuracy", "format_validity", "full_protocol_success", "trace_validity",
                    "capped_completions", "mean_output_tokens", "executor_calls",
                    "executor_failures", "policy_tokens", "executor_tokens",
                ):
                    value = summary.get(metric)
                    if isinstance(value, (int, float)):
                        scalar_logs[f"comparison/{arm}/{key}/{metric}"] = value
        for result in report["paired_comparisons"]:
            key = f"comparison/paired/{result['comparison']}/{result['split']}/{result['format']}"
            scalar_logs[f"{key}/accuracy_difference"] = result["accuracy_difference"]
            scalar_logs[f"{key}/ci_95_low"] = result["ci_95_low"]
            scalar_logs[f"{key}/ci_95_high"] = result["ci_95_high"]
        if scalar_logs:
            run.log(scalar_logs)
        run.summary.update(
            {
                "comparison/arms_present": report["arms_present"],
                "comparison/arms_missing": report["arms_missing"],
                "comparison/results_json": str(output_dir / "results.json"),
                "comparison/results_csv": str(output_dir / "results.csv"),
                "comparison/report_markdown": str(output_dir / "report.md"),
            }
        )
        for path in [
            output_dir / "results.json",
            output_dir / "results.csv",
            output_dir / "training_metrics.csv",
            output_dir / "report.md",
        ]:
            run.save(str(path))
        for path in report["figures"]:
            run.save(path)
        LOGGER.info(
            "Experiment 16 report written to %s; present=%s missing=%s",
            output_dir,
            report["arms_present"],
            report["arms_missing"],
        )
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
