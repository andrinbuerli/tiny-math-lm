from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from tinymath.checkpoint import load_checkpoint, save_checkpoint
from tinymath.config import (
    persist_resolved_config,
    read_saved_resolved_config,
    sanitized_config,
    workspace_path,
)
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


def test_experiment_presets_are_contiguously_numbered_and_documented():
    experiment_dir = Path(__file__).parents[1] / "configs" / "experiment"
    header_pattern = re.compile(
        r"^# Experiment (?P<number>\d{2}) — .+\n# Origin: .+$", re.MULTILINE
    )
    numbered_presets = []

    for path in experiment_dir.glob("*.yaml"):
        match = header_pattern.search(path.read_text())
        assert match is not None, f"{path.name} needs an experiment purpose and origin header"
        numbered_presets.append((int(match.group("number")), path.name))

    numbers = [number for number, _ in sorted(numbered_presets)]
    assert numbers == list(range(1, len(numbered_presets) + 1))


def test_resolved_config_is_sanitized_and_saved(workspace_tmpdir):
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(config_name="config", overrides=["+tracking.api_key=do-not-write"])
    values = sanitized_config(config)
    assert values["tracking"]["api_key"] == "<redacted>"
    path = persist_resolved_config(config, workspace_tmpdir)
    assert json.loads(path.read_text())["tracking"]["api_key"] == "<redacted>"
    assert read_saved_resolved_config(workspace_tmpdir)["tracking"]["project"] == "tinymath-rl"
    with pytest.raises(ValueError):
        workspace_path("..", "outside")


def test_sanitizer_preserves_tokenizer_and_completion_settings():
    values = sanitized_config(
        {
            "tokenizer_sha256": "public-hash",
            "rlvr": {"max_completion_tokens": 128},
            "tracking": {"wandb_api_key": "secret", "access_token": "secret"},
        }
    )
    assert values["tokenizer_sha256"] == "public-hash"
    assert values["rlvr"]["max_completion_tokens"] == 128
    assert values["tracking"]["wandb_api_key"] == "<redacted>"
    assert values["tracking"]["access_token"] == "<redacted>"


def test_seed_everything_repeats_torch_sequence():
    seed_everything(23)
    expected = torch.rand(5)
    seed_everything(23)
    assert torch.equal(torch.rand(5), expected)


def test_seed_everything_applies_configured_cpu_thread_cap():
    previous = torch.get_num_threads()
    try:
        seed_everything(23, cpu_threads=2)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous)


def test_checkpoint_round_trip(workspace_tmpdir):
    path = workspace_tmpdir / "state.pt"
    save_checkpoint(path, {"step": 9, "weights": torch.arange(4)})
    restored = load_checkpoint(path)
    assert restored["step"] == 9
    assert torch.equal(restored["weights"], torch.arange(4))


def test_wandb_offline_initialization_reuses_id(workspace_tmpdir, monkeypatch):
    captured = []

    class FakeRun:
        def __init__(self, run_id, mode):
            self.id = run_id
            self.settings = types.SimpleNamespace(mode=mode)
            self.summary = {}
            self.finished = False

        def finish(self):
            self.finished = True

    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        init=lambda **kwargs: captured.append(kwargs) or FakeRun(kwargs["id"], kwargs["mode"]),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(config_name="config", overrides=["tracking.mode=offline"])
    tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")
    first_id = captured[0]["id"]
    state = json.loads((workspace_tmpdir / "wandb_run.json").read_text())
    assert state["sync_status"] == "unsynced"
    second_run = tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")
    assert captured[1]["id"] == first_id
    assert captured[0]["mode"] == "offline"
    tracking.finish_wandb(second_run)
    assert second_run.finished
    assert json.loads((workspace_tmpdir / "wandb_run.json").read_text())["sync_status"] == "unsynced"
    config.tracking.name = "changed-config"
    with pytest.raises(tracking.TrackingError, match="different resolved Hydra config"):
        tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")


def test_wandb_resume_accepts_only_new_runtime_config_group(workspace_tmpdir, monkeypatch):
    captured = []

    class FakeRun:
        def __init__(self, run_id, mode):
            self.id = run_id
            self.settings = types.SimpleNamespace(mode=mode)
            self.summary = {}

        def finish(self):
            pass

    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        init=lambda **kwargs: captured.append(kwargs) or FakeRun(kwargs["id"], kwargs["mode"]),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        current = compose(config_name="config", overrides=["tracking.mode=offline"])
    saved = OmegaConf.create(current)
    del saved.runtime
    persist_resolved_config(saved, workspace_tmpdir)
    (workspace_tmpdir / "wandb_run.json").write_text(
        json.dumps({"id": "legacy-run", "mode": "offline", "sync_status": "unsynced"})
    )

    run = tracking.initialize_wandb(current, workspace_tmpdir, job_type="unit")
    assert run.id == "legacy-run"
    assert "runtime" in read_saved_resolved_config(workspace_tmpdir)
    tracking.finish_wandb(run)


def test_online_tracking_fails_clearly_without_credentials(workspace_tmpdir, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir="/workspace/configs"):
        config = compose(config_name="config")
    with pytest.raises(tracking.TrackingError, match="requires WANDB_API_KEY"):
        tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")


def test_workspace_credential_loader_accepts_legacy_assignment(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(tracking, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(
        tracking,
        "dotenv_values",
        lambda **kwargs: {"WANDBKEY": "unit-test-placeholder"},
    )
    tracking._load_workspace_credentials()
    assert os.environ["WANDB_API_KEY"] == "unit-test-placeholder"


def test_environment_credential_takes_precedence_over_legacy_assignment(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "environment-placeholder")
    monkeypatch.setattr(tracking, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(
        tracking,
        "dotenv_values",
        lambda **kwargs: pytest.fail("legacy file value should not be read"),
    )
    tracking._load_workspace_credentials()
    assert os.environ["WANDB_API_KEY"] == "environment-placeholder"
