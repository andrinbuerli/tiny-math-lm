"""Hydra orchestrator for shared data and matched SFT/GRPO/control runs."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tinymath.config import workspace_path
from tinymath.data.load import SELECTED_MODULES

LOGGER = logging.getLogger(__name__)
_GPU_LAUNCH_LOCK = threading.Lock()


def _gpu_load() -> tuple[int, int]:
    """Return utilization percent and free MiB for the single configured GPU."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first_gpu = result.stdout.splitlines()[0]
    utilization, free_mib = (int(value.strip()) for value in first_gpu.split(",", maxsplit=1))
    return utilization, free_mib


def _gpu_capacity_available(
    config: DictConfig, model: str, utilization: int, free_mib: int
) -> bool:
    """Require low active utilization and headroom for the next model."""
    runtime = config.runtime
    reserve_gb = float(runtime.parallel_model_vram_gb[model])
    headroom_gb = float(runtime.parallel_gpu_min_free_vram_gb)
    required_mib = (reserve_gb + headroom_gb) * 1024
    return (
        utilization <= int(runtime.parallel_gpu_max_utilization)
        and free_mib >= required_mib
    )


def _wait_for_gpu_capacity(config: DictConfig, model: str) -> None:
    while True:
        try:
            utilization, free_mib = _gpu_load()
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
            raise RuntimeError(
                "Cannot verify GPU utilization and VRAM before launching training or evaluation"
            ) from error
        if _gpu_capacity_available(config, model, utilization, free_mib):
            LOGGER.info(
                "GPU admits model=%s (utilization=%d%%, free_vram=%.1f GiB)",
                model,
                utilization,
                free_mib / 1024,
            )
            return
        LOGGER.info(
            "Waiting for GPU capacity for model=%s (utilization=%d%%, free_vram=%.1f GiB)",
            model,
            utilization,
            free_mib / 1024,
        )
        time.sleep(float(config.runtime.parallel_gpu_poll_seconds))


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
            if group == "tracking" and key == "tags":
                root_model_name = str(values["model"]["name"])
                value = [model if tag == root_model_name else tag for tag in value]
                if model not in value:
                    value.append(model)
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
    needs_gpu = script in {"train_sft.py", "train_rlvr.py", "evaluate.py"}
    if not needs_gpu:
        subprocess.run(command, cwd="/workspace", check=True)
        return
    # Serialize admission and allow telemetry to reflect each new process before
    # another worker considers starting a model on the same GPU.
    with _GPU_LAUNCH_LOCK:
        _wait_for_gpu_capacity(config, model)
        process = subprocess.Popen(command, cwd="/workspace")
        time.sleep(float(config.runtime.parallel_gpu_launch_settle_seconds))
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


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
