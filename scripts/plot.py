"""Hydra comparison report entry point with W&B provenance."""

from __future__ import annotations

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from tinymath.config import workspace_path
from tinymath.reporting import make_exp11_report, make_exp13_report, make_report
from tinymath.tracking.naming import comparison_identity
from tinymath.tracking.wandb import finish_wandb, initialize_wandb

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    experiment = config.experiment
    phase = str(experiment.get("phase", "main"))
    if int(experiment.get("number", 0) or 0) == 11 and phase in {"smoke", "pilot"}:
        examples = int(experiment.data_budget)
        updates = int(experiment.updates)
        run_dir = workspace_path(
            "runs",
            str(experiment.name),
            "comparison",
            f"{phase}-{examples}-examples-{updates}-updates",
            f"seed{int(config.seed)}",
        )
    else:
        run_dir = workspace_path(
            "runs", str(experiment.name), "comparison", f"seed{int(config.seed)}"
        )
    is_exp13 = int(experiment.get("number", 0) or 0) == 13
    if is_exp13:
        identity = comparison_identity(str(experiment.name), int(config.seed))
        config.tracking.name = identity["name"]
        config.tracking.group = identity["group"]
        config.tracking.tags = identity["tags"]
        run_dir = workspace_path(
            "runs", str(experiment.name), "comparison-report-resumed", f"seed{int(config.seed)}"
        )
    output_dir = workspace_path(str(config.reporting.output_dir))
    run = initialize_wandb(config, run_dir, job_type="comparison")
    try:
        if is_exp13:
            report, outputs = make_exp13_report(
                runs_dir=workspace_path("runs"),
                experiment=str(experiment.name),
                output_dir=output_dir,
                seed=int(config.seed),
                baseline_snapshot=workspace_path(str(config.reporting.baseline_snapshot)),
                annotation_bundle_dir=workspace_path(str(config.reporting.annotation_bundle_dir)),
                run_namespaces=OmegaConf.to_container(
                    config.reporting.get("run_namespaces", {}), resolve=True
                ),
                launch_blocker=(
                    str(experiment.launch_blocker)
                    if experiment.get("launch_blocker") else None
                ),
                launch_blockers=OmegaConf.to_container(
                    config.reporting.get("launch_blockers", {}), resolve=True
                ),
            )
            run.log({
                "comparison/rows": report["row_count"],
                "comparison/complete": sum(row["status"] == "complete" for row in report["rows"]),
                "comparison/partial": sum(row["status"] == "partial" for row in report["rows"]),
                "comparison/running": sum(row["status"] == "running" for row in report["rows"]),
                "comparison/not_run": sum(row["status"] == "not-run" for row in report["rows"]),
            })
        elif str(config.experiment.name) == "exp11_reasoning_500k_100k":
            report, outputs = make_exp11_report(
                runs_dir=workspace_path("runs"),
                experiment=str(config.experiment.name),
                output_dir=output_dir,
                seed=int(config.seed),
                baseline_snapshot=workspace_path(str(config.reporting.baseline_snapshot)),
                baseline_runs_dir=workspace_path("runs"),
                annotation_bundle_dir=(
                    workspace_path(str(config.reporting.annotation_bundle_dir))
                    if config.reporting.annotation_bundle_dir else None
                ),
            )
            run.log({
                "comparison/rows": report["row_count"],
                "comparison/complete": sum(row["exp11_status"] == "complete" for row in report["rows"]),
                "comparison/partial": sum(row["exp11_status"] == "partial" for row in report["rows"]),
            })
        else:
            report, outputs = make_report(
                runs_dir=workspace_path("runs"),
                experiment=str(config.experiment.name),
                output_dir=output_dir,
                seed=int(config.seed),
                data_bundle_dir=(
                    workspace_path(str(config.data_bundle)) if config.data_bundle else None
                ),
            )
            run.log({"comparison/available_evaluations": report["coverage"]["available_evaluations"]})
        for artifact in outputs:
            if artifact.suffix == ".png":
                run.log({f"comparison/{artifact.stem}": wandb.Image(str(artifact))})
        report_artifact = wandb.Artifact(
            name=f"{config.experiment.name}-comparison-seed{int(config.seed)}",
            type="comparison-report",
            metadata={
                "experiment": str(config.experiment.name),
                "seed": int(config.seed),
                "available_evaluations": (
                    report["coverage"]["available_evaluations"]
                    if "coverage" in report else sum(row.get("status") == "complete" for row in report["rows"])
                ),
            },
        )
        for artifact_path in outputs:
            report_artifact.add_file(str(artifact_path))
        run.log_artifact(report_artifact)
        run.summary.update(
            {
                "comparison/report_path": str(output_dir / "comparison.json"),
                "comparison/markdown_path": str(output_dir / (
                    "intermediate_results.md"
                    if str(config.experiment.name) == "exp11_reasoning_500k_100k"
                    or int(experiment.get("number", 0) or 0) == 13
                    else "report.md"
                )),
                "comparison/csv_path": str(output_dir / "comparison.csv"),
                "comparison/available_evaluations": (
                    report["coverage"]["available_evaluations"]
                    if "coverage" in report else sum(row.get("status") == "complete" for row in report["rows"])
                ),
            }
        )
        LOGGER.info("Comparison report written to %s", output_dir)
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
