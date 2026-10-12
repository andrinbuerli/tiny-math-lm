"""Reproducibility and runtime environment helpers."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    torch: str
    numpy: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    git_commit: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch, including CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def runtime_info() -> RuntimeInfo:
    """Capture versions and the local Git revision without contacting a remote."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    cuda_available = torch.cuda.is_available()
    return RuntimeInfo(
        python=sys.version.split()[0],
        torch=torch.__version__,
        numpy=np.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0) if cuda_available else None,
        git_commit=commit,
    )
