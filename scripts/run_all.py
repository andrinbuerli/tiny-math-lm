"""Hydra orchestrator for shared data and matched SFT/GRPO/control runs."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tinymath.config import workspace_path
from tinymath.data.load import SELECTED_MODULES

LOGGER = logging.getLogger(__name__)


def _overrides(
    config: DictConfig,
    *,
    model: str,
    trainer: str,
    experiment_choice: str | None = None,
    checkpoint: str | None = None,
    init_checkpoint: str | None = None,
) -> list[str]:
    values = OmegaConf.to_container(config, resolve=True)
    assert isinstance(values, dict)
    if experiment_choice is None:
        experiment_choice = HydraConfig.get().runtime.choices.get(
            "experiment", values["experiment"]["name"]
        )
    result = [
        f"model={model}", f"trainer={trainer}",
        f"experiment={experiment_choice}",
        f"seed={values['seed']}",
    ]
    for key, value in values["experiment"].items():
        if key != "name":
            result.append(f"experiment.{key}={json.dumps(value)}")
    for group in ("data", "rlvr", "eval", "qualitative", "runtime", "tracking", "paths"):
        for key, value in values[group].items():
            if key == "name":
                continue
            if group == "data" and key == "modules" and set(value) != set(SELECTED_MODULES):
                raise ValueError("The orchestration requires exactly the five specified modules")
            result.append(f"{group}.{key}={json.dumps(value)}")
    for key in ("init_checkpoint", "data_bundle"):
        value = values.get(key)
        if value is not None:
            result.append(f"{key}={json.dumps(value)}")
    if checkpoint is not None:
        result.append(f"checkpoint={json.dumps(checkpoint)}")
    if init_checkpoint is not None:
        result.append(f"init_checkpoint={json.dumps(init_checkpoint)}")
    return result


def _run(
    script: str,
    config: DictConfig,
    *,
    model: str,
    trainer: str,
    checkpoint: str | None = None,
    init_checkpoint: str | None = None,
    experiment_choice: str | None = None,
) -> None:
    command = [
        str(workspace_path(".tools", "uv")), "run", "--locked", "python",
        str(workspace_path("scripts", script)),
        *_overrides(
            config,
            model=model,
            trainer=trainer,
            experiment_choice=experiment_choice,
            checkpoint=checkpoint,
            init_checkpoint=init_checkpoint,
        ),
    ]
    LOGGER.info("Running %s model=%s trainer=%s", script, model, trainer)
    subprocess.run(command, cwd="/workspace", check=True)


def _run_model(config: DictConfig, model_name: str, experiment_choice: str) -> None:
    """Run a model's SFT, GRPO and continued-SFT arms and their evaluations."""
    _run(
        "train_sft.py", config, model=model_name, trainer="sft",
        experiment_choice=experiment_choice,
    )
    sft_run = workspace_path(
        "runs", str(config.experiment.name), model_name, "sft", f"seed{int(config.seed)}"
    )
    sft_checkpoint = str(sft_run / "best.pt")
    _run(
        "evaluate.py", config, model=model_name, trainer="sft", checkpoint=sft_checkpoint,
        experiment_choice=experiment_choice,
    )
    if config.orchestration.run_grpo:
        _run(
            "train_rlvr.py", config, model=model_name, trainer="sft",
            init_checkpoint=sft_checkpoint, experiment_choice=experiment_choice,
        )
        grpo_checkpoint = workspace_path(
            "runs", str(config.experiment.name), model_name, "grpo", f"seed{int(config.seed)}",
            "latest.pt",
        )
        _run(
            "evaluate.py", config, model=model_name, trainer="sft",
            checkpoint=str(grpo_checkpoint), experiment_choice=experiment_choice,
        )
    if config.orchestration.run_continued_sft:
        control_run = workspace_path(
            "runs", str(config.experiment.name), model_name, "continued_sft", f"seed{int(config.seed)}"
        )
        _run(
            "train_sft.py", config, model=model_name, trainer="continued_sft",
            init_checkpoint=sft_checkpoint, experiment_choice=experiment_choice,
        )
        _run(
            "evaluate.py", config, model=model_name, trainer="sft",
            checkpoint=str(control_run / "best.pt"), experiment_choice=experiment_choice,
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if not config.experiment.run_optimization:
        if not config.data_bundle:
            _run("prepare_data.py", config, model=str(config.model.name), trainer="sft")
        LOGGER.info("Experiment preset requested data preparation only")
        return
    selected_models: Iterable[str] = config.orchestration.models
    if config.data_bundle:
        manifest = workspace_path(str(config.data_bundle)) / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Prepared data manifest not found: {manifest}")
    else:
        _run("prepare_data.py", config, model=str(config.model.name), trainer="sft")
    experiment_choice = HydraConfig.get().runtime.choices.get(
        "experiment", str(config.experiment.name)
    )
    model_names = [str(name) for name in selected_models]
    if config.orchestration.get("parallel_models", False) and len(model_names) > 1:
        with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
            futures = [
                executor.submit(_run_model, config, name, experiment_choice)
                for name in model_names
            ]
            for future in futures:
                future.result()
    else:
        for model_name in model_names:
            _run_model(config, model_name, experiment_choice)
    _run("plot.py", config, model=str(config.model.name), trainer="sft")


if __name__ == "__main__":
    main()
