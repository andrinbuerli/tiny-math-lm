from __future__ import annotations

import json
import sys
import types

import pytest
import torch
from hydra import compose, initialize_config_dir

from tinymath.checkpoint import load_checkpoint, save_checkpoint
from tinymath.config import persist_resolved_config, sanitized_config, workspace_path
from tinymath.repro import seed_everything
from tinymath.tracking import wandb as tracking


def test_hydra_composes_all_experiment_groups_and_overrides():
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(
            config_name="config",
            overrides=["model=10m", "experiment=pilot", "tracking.project=unit-test", "seed=23"],
        )
    assert config.model.name == "10m"
    assert config.experiment.data_budget == 20000
    assert config.tracking.project == "unit-test"
    assert config.tracking.name == "pilot-10m-seed23"


def test_resolved_config_is_sanitized_and_saved(workspace_tmpdir):
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(config_name="config", overrides=["+tracking.api_key=do-not-write"])
    values = sanitized_config(config)
    assert values["tracking"]["api_key"] == "<redacted>"
    path = persist_resolved_config(config, workspace_tmpdir)
    assert json.loads(path.read_text())["tracking"]["api_key"] == "<redacted>"
    with pytest.raises(ValueError):
        workspace_path("..", "outside")


def test_seed_everything_repeats_torch_sequence():
    seed_everything(23)
    expected = torch.rand(5)
    seed_everything(23)
    assert torch.equal(torch.rand(5), expected)


def test_checkpoint_round_trip(workspace_tmpdir):
    path = workspace_tmpdir / "state.pt"
    save_checkpoint(path, {"step": 9, "weights": torch.arange(4)})
    restored = load_checkpoint(path)
    assert restored["step"] == 9
    assert torch.equal(restored["weights"], torch.arange(4))


def test_wandb_offline_initialization_reuses_id(workspace_tmpdir, monkeypatch):
    captured = []
    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        init=lambda **kwargs: captured.append(kwargs) or "run",
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(config_name="config", overrides=["tracking.mode=offline"])
    assert tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit") == "run"
    first_id = captured[0]["id"]
    assert tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit") == "run"
    assert captured[1]["id"] == first_id
    assert captured[0]["mode"] == "offline"


def test_online_tracking_fails_clearly_without_credentials(workspace_tmpdir, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(config_name="config")
    with pytest.raises(tracking.TrackingError, match="requires WANDB_API_KEY"):
        tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")
