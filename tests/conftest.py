from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def workspace_tmpdir():
    path = Path(tempfile.mkdtemp(prefix="tinymath-test-", dir="/workspace"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
