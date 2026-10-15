"""Hydra orchestrator for shared data and matched SFT/GRPO/control runs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
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
from tinymath.generation import RolloutConfig
from tinymath.train.budgets import isolated_training_run_dir

LOGGER = logging.getLogger(__name__)
_GPU_LAUNCH_LOCK = threading.Lock()
EXP13_CANONICAL_BUNDLE = "data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18"
EXP13_CANONICAL_MANIFEST_SHA256 = "5d493a0f4c961b84b873036ea1bc93ebb6d01b6f022d7ed81997b094da3a018f"


def _hydra_value(value: object) -> str:
    """Serialize resolved values into Hydra override grammar."""
    if isinstance(value, dict):
        entries = ",".join(f"{key}:{_hydra_value(item)}" for key, item in value.items())
        return "{" + entries + "}"
    return json.dumps(value)


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
            LOGGER.warning(
                "Cannot verify GPU utilization and VRAM for model=%s; "
                "waiting before any GPU launch (%s)",
                model,
                error,
            )
            time.sleep(float(config.runtime.parallel_gpu_poll_seconds))
            continue
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
    run_arm: str | None = None,
    run_label: str | None = None,
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
        f"experiment.name={_hydra_value(values['experiment']['name'])}",
        f"seed={values['seed']}",
    ]
    for key, value in values["experiment"].items():
        if key != "name":
            # A literal backslash-u in a saved title must survive Hydra's
            # parsing unchanged when this resolved config is forwarded again.
            if key == "title" and isinstance(value, str) and "\\" in value and "'" not in value:
                serialized = f"'{value}'"
            else:
                serialized = json.dumps(value)
            result.append(f"experiment.{key}={serialized}")
    for group in (
        "data", "rlvr", "rollout", "eval", "qualitative", "runtime",
        "reporting", "tracking", "paths",
    ):
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
            result.append(f"{group}.{key}={_hydra_value(value)}")
    for key in ("init_checkpoint", "data_bundle"):
        value = values.get(key)
        if value is not None:
            result.append(f"{key}={_hydra_value(value)}")
    if checkpoint is not None:
        result.append(f"checkpoint={_hydra_value(checkpoint)}")
    if init_checkpoint is not None:
        result.append(f"init_checkpoint={_hydra_value(init_checkpoint)}")
    if run_arm is not None:
        experiment_name = str(values["experiment"]["name"])
        run_name = (
            f"{experiment_name}-{model}-{run_arm}-{run_label or trainer}-"
            f"seed{int(values['seed'])}"
        )
        run_group = f"{experiment_name}-{model}-{run_arm}"
        run_tags = [experiment_name, model, run_arm]
        result.extend([
            f"tracking.name={_hydra_value(run_name)}",
            f"tracking.group={_hydra_value(run_group)}",
            f"tracking.tags={_hydra_value(run_tags)}",
        ])
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
    run_arm: str | None = None,
    run_label: str | None = None,
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
            run_arm=run_arm,
            run_label=run_label,
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


def _downstream_config(config: DictConfig) -> DictConfig:
    """Upgrade legacy resumed reasoning-v3 configs for new downstream runs."""
    if str(config.experiment.get("protocol", "")) == "reasoning_v3":
        # Resumed SFT configs can carry a legacy rollout schema that was used
        # only for qualitative logging. New reasoning-v3 arms and evaluations
        # must use the current exact v3 rollout/scorer identity.
        downstream_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        downstream_config.rollout = RolloutConfig(executor="exact").identity()
        return downstream_config
    return config


def _evaluation_config(config: DictConfig) -> DictConfig:
    """Evaluate Experiment 13 on the verified canonical reasoning-v3 bundle."""
    if int(config.experiment.get("number", 0) or 0) != 13:
        return config
    bundle_dir = workspace_path(EXP13_CANONICAL_BUNDLE)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical Experiment 13 bundle not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha256 != EXP13_CANONICAL_MANIFEST_SHA256:
        raise ValueError(
            "Experiment 13 evaluation bundle manifest does not match the verified canonical bundle: "
            f"expected {EXP13_CANONICAL_MANIFEST_SHA256}, got {manifest_sha256}"
        )
    protocol = manifest.get("annotation", {}).get("protocol")
    if protocol != "reasoning_v3":
        raise ValueError(
            f"Experiment 13 evaluation requires a reasoning_v3 bundle, got {protocol!r}"
        )
    evaluation_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    evaluation_config.data_bundle = EXP13_CANONICAL_BUNDLE
    evaluation_config.reporting.annotation_bundle_dir = str(bundle_dir)
    return evaluation_config


def _run_model(config: DictConfig, model_name: str, experiment_choice: str) -> None:
    """Run a model's SFT, GRPO and continued-SFT arms and their evaluations."""
    downstream_config = _downstream_config(config)
    evaluation_config = _evaluation_config(downstream_config)
    _run(
        "train_sft.py", config, model=model_name, trainer="sft",
        experiment_choice=experiment_choice,
        run_arm="sft", run_label="training",
    )
    sft_run = workspace_path(
        isolated_training_run_dir(
            config,
            model=model_name,
            arm="sft",
            runs_root=str(workspace_path("runs")),
        )
    )
    sft_checkpoint = str(sft_run / "best.pt")
    _run(
        "evaluate.py", evaluation_config, model=model_name, trainer="sft", checkpoint=sft_checkpoint,
        experiment_choice=experiment_choice,
        run_arm="sft", run_label="evaluation",
    )
    if config.orchestration.run_grpo:
        _run(
            "train_rlvr.py", downstream_config, model=model_name, trainer="sft",
            init_checkpoint=sft_checkpoint, experiment_choice=experiment_choice,
            run_arm="grpo", run_label="training",
        )
        grpo_checkpoint = workspace_path(
            isolated_training_run_dir(
                config,
                model=model_name,
                arm="grpo",
                runs_root=str(workspace_path("runs")),
            ),
            "latest.pt",
        )
        _run(
            "evaluate.py", evaluation_config, model=model_name, trainer="sft",
            checkpoint=str(grpo_checkpoint), experiment_choice=experiment_choice,
            run_arm="grpo", run_label="evaluation",
        )
    if config.orchestration.run_continued_sft:
        control_run = workspace_path(
            isolated_training_run_dir(
                config,
                model=model_name,
                arm="continued_sft",
                runs_root=str(workspace_path("runs")),
            )
        )
        _run(
            "train_sft.py", downstream_config, model=model_name, trainer="continued_sft",
            init_checkpoint=sft_checkpoint, experiment_choice=experiment_choice,
            run_arm="continued_sft", run_label="training",
        )
        _run(
            "evaluate.py", evaluation_config, model=model_name, trainer="sft",
            checkpoint=str(control_run / "best.pt"), experiment_choice=experiment_choice,
            run_arm="continued_sft", run_label="evaluation",
        )


def _run_comparison(config: DictConfig) -> None:
    if os.environ.get("TINYMATH_SKIP_COMPARISON") != "1":
        _run("plot.py", config, model=str(config.model.name), trainer="sft")


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
    _run_comparison(config)


if __name__ == "__main__":
    main()
