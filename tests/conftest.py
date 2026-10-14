from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def workspace_tmpdir():
    # Keep transient test files under the checked-out project, wherever the
    # checkout is mounted (the managed GPU container uses /workspace).
    project_root = Path(__file__).resolve().parents[1]
    path = Path(tempfile.mkdtemp(prefix="tinymath-test-", dir=project_root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
