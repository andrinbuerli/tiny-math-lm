"""Single-GPU admission checks shared by Experiment 16 HF entry points."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

LOGGER = logging.getLogger(__name__)


def gpu_load() -> tuple[int, int]:
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
    utilization, free_mib = (
        int(value.strip()) for value in result.stdout.splitlines()[0].split(",", maxsplit=1)
    )
    return utilization, free_mib


def gpu_capacity_available(config: Any, model: str, utilization: int, free_mib: int) -> bool:
    runtime = config.runtime
    reserve_gb = float(runtime.parallel_model_vram_gb[model])
    headroom_gb = float(runtime.parallel_gpu_min_free_vram_gb)
    return (
        utilization <= int(runtime.parallel_gpu_max_utilization)
        and free_mib >= (reserve_gb + headroom_gb) * 1024
    )


def wait_for_gpu_capacity(config: Any, model: str) -> None:
    while True:
        try:
            utilization, free_mib = gpu_load()
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
            LOGGER.warning("Cannot verify GPU state; waiting before loading %s (%s)", model, error)
            time.sleep(float(config.runtime.parallel_gpu_poll_seconds))
            continue
        if gpu_capacity_available(config, model, utilization, free_mib):
            LOGGER.info(
                "GPU admits %s: utilization=%d%% free_vram=%.2f GiB",
                model,
                utilization,
                free_mib / 1024,
            )
            return
        LOGGER.info(
            "Waiting for GPU capacity for %s: utilization=%d%% free_vram=%.2f GiB",
            model,
            utilization,
            free_mib / 1024,
        )
        time.sleep(float(config.runtime.parallel_gpu_poll_seconds))
