"""Random seeding and local runtime provenance."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
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
    uv_lock_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and available CUDA generators."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def runtime_info(workspace: str | Path = "/workspace") -> RuntimeInfo:
    lockfile = Path(workspace) / "uv.lock"
    lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest() if lockfile.exists() else None
    available = torch.cuda.is_available()
    return RuntimeInfo(
        python=sys.version.split()[0],
        torch=torch.__version__,
        numpy=np.__version__,
        cuda_available=available,
        cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name(0) if available else None,
        git_commit=_git_commit(),
        uv_lock_sha256=lock_hash,
    )
