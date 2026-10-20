from __future__ import annotations

import importlib.util
from pathlib import Path

from omegaconf import OmegaConf

_SPEC = importlib.util.spec_from_file_location(
    "run_exp16", Path(__file__).resolve().parents[1] / "scripts" / "run_exp16.py"
)
assert _SPEC and _SPEC.loader
run_exp16 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_exp16)


def test_exp16_child_command_uses_hydra_group_not_run_identity(monkeypatch):
    observed = {}
    monkeypatch.setattr(run_exp16.shutil, "which", lambda _name: "/workspace/.tools/uv")
    monkeypatch.setattr(
        run_exp16.subprocess,
        "run",
        lambda command, **kwargs: observed.update(command=command, **kwargs),
    )
    config = OmegaConf.create(
        {"experiment": {"config_group": "16_qwen3_base_reasoning_v3", "name": "exp16_qwen3_base_reasoning_v3"}}
    )

    run_exp16._run(config, "train_sft", "experiment.phase=pilot")

    assert observed["command"][5] == "experiment=16_qwen3_base_reasoning_v3"
    assert observed["command"][6] == "experiment.phase=pilot"
    assert observed["cwd"] == "/workspace"
    assert observed["check"] is True
