from __future__ import annotations

import json
import os
import re
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
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
from tinymath.tracking.provenance import read_wandb_run_id


def test_hydra_composes_all_experiment_groups_and_overrides():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="config",
            overrides=["model=10m", "experiment=02_pilot", "tracking.project=unit-test", "seed=23"],
        )
    assert config.model.name == "10m"
    assert config.experiment.data_budget == 20000
    assert config.data.revision == "d9f78d0665b3a13dad21c6830b9b358c8a25ae74"
    assert config.tracking.project == "unit-test"
    assert config.tracking.name == "pilot-10m-seed23"
    assert config.orchestration.parallel_models is False
    script = Path(__file__).parents[1] / "scripts" / "run_all.py"
    spec = spec_from_file_location("tinymath_run_all", script)
    assert spec is not None and spec.loader is not None
    run_all = module_from_spec(spec)
    spec.loader.exec_module(run_all)
    overrides = run_all._overrides(
        config,
        model="100m",
        trainer="sft",
        experiment_choice="02_pilot",
    )
    assert "experiment=02_pilot" in overrides
    assert "model=100m" in overrides
    assert 'tracking.tags=["pilot", "100m"]' in overrides


def test_parallel_gpu_admission_requires_utilization_and_vram_headroom():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="config")
    script = Path(__file__).parents[1] / "scripts" / "run_all.py"
    spec = spec_from_file_location("tinymath_run_all_gpu", script)
    assert spec is not None and spec.loader is not None
    run_all = module_from_spec(spec)
    spec.loader.exec_module(run_all)

    assert run_all._gpu_capacity_available(config, "100m", 75, 10 * 1024)
    assert not run_all._gpu_capacity_available(config, "100m", 76, 20 * 1024)
    assert not run_all._gpu_capacity_available(config, "100m", 50, 10 * 1024 - 1)


def test_wandb_go_service_thread_limit_respects_runtime_and_environment(monkeypatch):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="config", overrides=["runtime.wandb_service_gomaxprocs=3"])

    monkeypatch.delenv("GOMAXPROCS", raising=False)
    tracking._configure_wandb_service_threads(config)
    assert os.environ["GOMAXPROCS"] == "3"

    monkeypatch.setenv("GOMAXPROCS", "1")
    tracking._configure_wandb_service_threads(config)
    assert os.environ["GOMAXPROCS"] == "1"


def test_resume_accepts_wandb_thread_limit_runtime_extension():
    saved = {"runtime": {"cpu_threads": 4}}
    current = {"runtime": {"cpu_threads": 4, "wandb_service_gomaxprocs": 2}}
    assert tracking._compatible_with_runtime_extension(saved, current)

    changed_training_runtime = {
        "runtime": {"cpu_threads": 8, "wandb_service_gomaxprocs": 2}
    }
    assert not tracking._compatible_with_runtime_extension(saved, changed_training_runtime)


def test_resume_allows_wandb_mode_change_without_training_config_changes():
    saved = {
        "tracking": {"mode": "offline", "entity": None},
        "trainer": {"max_steps": 100},
    }
    current = {
        "tracking": {"mode": "online", "entity": "verified-entity"},
        "trainer": {"max_steps": 100},
    }
    assert tracking._compatible_with_runtime_extension(saved, current)

    changed_training = {"tracking": {"mode": "online"}, "trainer": {"max_steps": 101}}
    assert not tracking._compatible_with_runtime_extension(saved, changed_training)


def test_resume_allows_reasoning_generation_ceiling_to_match_context():
    saved = {
        "experiment": {"protocol": "reasoning_v1"},
        "model": {"context_length": 512},
        "eval": {"greedy": True, "max_new_tokens": 128},
    }
    current = {
        "experiment": {"protocol": "reasoning_v1"},
        "model": {"context_length": 512},
        "eval": {"greedy": True, "max_new_tokens": 512},
    }
    assert tracking._compatible_with_runtime_extension(saved, current)

    changed_context = {**current, "model": {"context_length": 256}}
    assert not tracking._compatible_with_runtime_extension(saved, changed_context)


def test_resume_allows_only_reasoning_v3_sampler_strategy_change():
    saved = {
        "experiment": {"protocol": "reasoning_v3"},
        "data_bundle": "bundle-a",
        "rollout": {
            "rollout_version": 3,
            "scorer_version": 3,
            "executor_version": 1,
        },
    }
    current = {
        "experiment": {"protocol": "reasoning_v3"},
        "data_bundle": "bundle-a",
        "rollout": {
            "rollout_version": 3,
            "scorer_version": 3,
            "executor_version": 1,
            "sampler_version": 2,
            "prompt_batching": "left_pad",
        },
    }
    assert tracking._compatible_with_runtime_extension(saved, current)

    changed_scorer = {
        **current,
        "rollout": {**current["rollout"], "scorer_version": 4},
    }
    assert not tracking._compatible_with_runtime_extension(saved, changed_scorer)

    changed_bundle = {**current, "data_bundle": "bundle-b"}
    assert not tracking._compatible_with_runtime_extension(saved, changed_bundle)


def test_wandb_entity_defaults_to_configured_team(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="config",
            overrides=["experiment=11_reasoning_500k_100k", "model=1m"],
        )
    assert config.tracking.entity == "andi-mueller-csem-sa"


def test_continued_sft_config_approximately_matches_grpo_compute():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="config",
            overrides=["trainer=continued_sft", "experiment=10_restarted_100k_greedy"],
        )
    assert config.trainer.max_steps == 100000
    assert config.trainer.batch_size == 32
    assert config.trainer.gradient_accumulation == 4


def test_experiment_presets_are_contiguously_numbered_and_documented():
    experiment_dir = Path(__file__).parents[1] / "configs" / "experiment"
    header_pattern = re.compile(
        r"^# Experiment (?P<number>\d{2}) — .+\n# Origin: .+$", re.MULTILINE
    )
    numbered_presets = []

    for path in experiment_dir.glob("*.yaml"):
        match = header_pattern.search(path.read_text())
        assert match is not None, f"{path.name} needs an experiment purpose and origin header"
        assert path.stem.startswith(f"{match.group('number')}_"), (
            f"{path.name} must include its documented experiment number"
        )
        numbered_presets.append((int(match.group("number")), path.name))

    numbers = [number for number, _ in sorted(numbered_presets)]
    annotation_dir = experiment_dir.parent / "annotation"
    annotation_only_numbers = {
        int(config.experiment.number)
        for path in annotation_dir.glob("*.yaml")
        if (config := OmegaConf.load(path)).get("experiment")
        and config.experiment.get("number") is not None
        and not any(experiment_dir.glob(f"{int(config.experiment.number):02d}_*.yaml"))
    }
    expected = [number for number in range(1, max(numbers) + 1) if number not in annotation_only_numbers]
    assert numbers == expected


def test_resolved_config_is_sanitized_and_saved(workspace_tmpdir):
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
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


def test_read_wandb_run_id_from_local_run_metadata(workspace_tmpdir):
    run_dir = workspace_tmpdir / "training"
    run_dir.mkdir()
    assert read_wandb_run_id(run_dir) is None
    (run_dir / "wandb_run.json").write_text(
        json.dumps({"id": "run-123", "mode": "online"}), encoding="utf-8"
    )
    assert read_wandb_run_id(run_dir) == "run-123"
    (run_dir / "wandb_run.json").write_text("not json", encoding="utf-8")
    assert read_wandb_run_id(run_dir) is None


def test_wandb_offline_initialization_reuses_id(workspace_tmpdir, monkeypatch):
    captured = []

    class FakeRun:
        def __init__(self, run_id, mode):
            self.id = run_id
            self.settings = types.SimpleNamespace(mode=mode)
            self.summary = {}
            self.config = types.SimpleNamespace(update=lambda *_args, **_kwargs: None)
            self.finished = False

        def finish(self, *, exit_code=None):
            self.finished = True
            self.exit_code = exit_code

    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        errors=types.SimpleNamespace(Error=RuntimeError),
        init=lambda **kwargs: captured.append(kwargs) or FakeRun(kwargs["id"], kwargs["mode"]),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
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


def test_exp11_wandb_uses_canonical_group_and_training_name(workspace_tmpdir, monkeypatch):
    captured = []

    class FakeRun:
        id = "stable-run"
        url = "https://wandb.ai/unit/tinymath-rl/runs/stable-run"
        settings = types.SimpleNamespace(mode="offline")

        def __init__(self):
            self.summary = {}
            self.config = types.SimpleNamespace(update=lambda *_args, **_kwargs: None)

        def finish(self, *, exit_code=None):
            self.exit_code = exit_code

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(
            Settings=lambda **kwargs: kwargs,
            errors=types.SimpleNamespace(Error=RuntimeError),
            init=lambda **kwargs: captured.append(kwargs) or FakeRun(),
        ),
    )
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="config",
            overrides=["experiment=11_reasoning_500k_100k", "model=10m", "tracking.mode=offline"],
        )

    run_dir = workspace_tmpdir / "exp11-grpo"
    tracking.initialize_wandb(config, run_dir, job_type="grpo", job_metadata={"phase": "main"})

    assert captured[0]["project"] == "tinymath-rl"
    assert captured[0]["group"] == "exp11-reasoning-500k-100k-seed18"
    assert captured[0]["name"] == "exp11-reasoning-500k-100k-main-10m-grpo-seed18"
    state = json.loads((run_dir / "wandb_run.json").read_text())
    assert state["identity"]["name"] == captured[0]["name"]
    assert state["url"] == FakeRun.url
    assert state["project"] == "tinymath-rl"


def test_online_sync_status_requires_successful_finish(workspace_tmpdir):
    class FakeRun:
        id = "online-run"
        settings = types.SimpleNamespace(mode="online")

        def __init__(self, *, interrupt=False):
            self.interrupt = interrupt
            self.summary = {}

        def finish(self, *, exit_code=None):
            self.exit_code = exit_code
            if self.interrupt:
                raise KeyboardInterrupt

    for interrupt, timed_out, expected in (
        (True, False, "pending"),
        (False, True, "pending"),
        (False, False, "synced"),
    ):
        run_dir = workspace_tmpdir / str(interrupt) / str(timed_out)
        run_dir.mkdir(parents=True)
        state_path = run_dir / "wandb_run.json"
        state_path.write_text(
            json.dumps({"id": "online-run", "mode": "online", "sync_status": "pending"})
        )
        tracking._ACTIVE_STATE_PATHS["online-run"] = state_path
        run = FakeRun(interrupt=interrupt)
        run._finish_timed_out = timed_out
        if interrupt:
            with pytest.raises(KeyboardInterrupt):
                tracking.finish_wandb(run)
        else:
            tracking.finish_wandb(run)
        state = json.loads(state_path.read_text())
        assert state["sync_status"] == expected
        assert state["run_status"] == ("failed" if interrupt else "finished")


def test_failed_run_is_finished_with_nonzero_exit_code(workspace_tmpdir):
    class FakeRun:
        id = "failed-run"
        settings = types.SimpleNamespace(mode="online")
        _finish_timed_out = False

        def __init__(self):
            self.summary = {}
            self.exit_code = None

        def finish(self, *, exit_code=None):
            self.exit_code = exit_code

    state_path = workspace_tmpdir / "wandb_run.json"
    state_path.write_text(
        json.dumps({"id": "failed-run", "mode": "online", "sync_status": "pending"})
    )
    tracking._ACTIVE_STATE_PATHS["failed-run"] = state_path
    run = FakeRun()
    tracking.finish_wandb(run, exit_code=1)

    state = json.loads(state_path.read_text())
    assert run.exit_code == 1
    assert state["id"] == "failed-run"
    assert state["sync_status"] == "synced"
    assert state["run_status"] == "failed"
    assert state["exit_code"] == 1


def test_tracking_state_write_failure_preserves_existing_id(workspace_tmpdir, monkeypatch):
    path = workspace_tmpdir / "wandb_run.json"
    original = json.dumps({"id": "preserved-run", "mode": "online"})
    path.write_text(original, encoding="utf-8")

    def fail_replace(_src, _dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(tracking.os, "replace", fail_replace)
    with pytest.raises(OSError, match="no space left"):
        tracking._write_tracking_state(path, {"id": "new-run"})

    assert path.read_text(encoding="utf-8") == original
    assert not path.with_name("wandb_run.json.tmp").exists()


def test_wandb_initialization_failure_preserves_id_and_marks_failed(
    workspace_tmpdir, monkeypatch
):
    class WandbError(Exception):
        pass

    def fail_init(**_kwargs):
        raise WandbError("simulated service startup failure")

    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        errors=types.SimpleNamespace(Error=WandbError),
        init=fail_init,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="config", overrides=["tracking.mode=offline"])

    with pytest.raises(tracking.TrackingError, match="simulated service startup failure"):
        tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")

    state = json.loads((workspace_tmpdir / "wandb_run.json").read_text())
    assert state["id"]
    assert state["run_status"] == "failed"
    assert state["exit_code"] == 1
    assert state["sync_status"] == "unsynced"


def test_wandb_resume_accepts_safe_config_extensions(workspace_tmpdir, monkeypatch):
    captured = []

    class FakeRun:
        def __init__(self, run_id, mode):
            self.id = run_id
            self.settings = types.SimpleNamespace(mode=mode)
            self.summary = {}
            self.config = types.SimpleNamespace(update=lambda *_args, **_kwargs: None)

        def finish(self, *, exit_code=None):
            pass

    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        init=lambda **kwargs: captured.append(kwargs) or FakeRun(kwargs["id"], kwargs["mode"]),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        current = compose(
            config_name="config",
            overrides=[
                "tracking.mode=offline",
                "data_bundle=data/processed/example-bundle",
            ],
        )
    saved = OmegaConf.create(current)
    del saved.runtime
    del saved.eval.temperature
    saved.data.revision = None
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
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
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


def test_wandb_starts_before_cuda_runtime_metadata(workspace_tmpdir, monkeypatch):
    events = []

    class FakeConfig:
        def update(self, values, *, allow_val_change=False):
            events.append(("config", values, allow_val_change))

    class FakeRun:
        id = "ordered-run"
        settings = types.SimpleNamespace(mode="offline")

        def __init__(self):
            self.config = FakeConfig()
            self.summary = {}

        def finish(self, *, exit_code=None):
            events.append(("finish", exit_code))

    def fake_init(**kwargs):
        events.append(("init", kwargs["config"]))
        return FakeRun()

    fake_wandb = types.SimpleNamespace(
        Settings=lambda **kwargs: kwargs,
        errors=types.SimpleNamespace(Error=RuntimeError),
        init=fake_init,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(tracking, "_load_workspace_credentials", lambda: None)
    monkeypatch.setattr(
        tracking,
        "runtime_info",
        lambda: events.append(("runtime_info",)) or types.SimpleNamespace(as_dict=lambda: {"cuda": True}),
    )
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="config", overrides=["tracking.mode=offline"])

    run = tracking.initialize_wandb(config, workspace_tmpdir, job_type="unit")
    assert [event[0] for event in events] == ["init", "runtime_info", "config"]
    assert "runtime" not in events[0][1]
    assert events[2][1] == {"runtime": {"cuda": True}}
    tracking.finish_wandb(run)


def test_tracked_run_initializes_wandb_before_seeding_and_finishes_on_seed_error(monkeypatch):
    events = []
    run = object()
    config = types.SimpleNamespace(seed=29, runtime=types.SimpleNamespace(cpu_threads=1))
    monkeypatch.setattr(
        tracking,
        "initialize_wandb",
        lambda current, run_dir, *, job_type, job_metadata=None: events.append(
            ("wandb", job_type, job_metadata)
        )
        or run,
    )
    monkeypatch.setattr(
        tracking,
        "seed_everything",
        lambda seed, *, cpu_threads: events.append(("seed", seed, cpu_threads)),
    )
    assert tracking.initialize_tracked_run(config, "/tmp/run", job_type="unit") is run
    assert events == [("wandb", "unit", None), ("seed", 29, 1)]

    events.clear()

    def fail_seed(seed, *, cpu_threads):
        events.append(("seed", seed, cpu_threads))
        raise RuntimeError("seeding failed")

    monkeypatch.setattr(tracking, "seed_everything", fail_seed)
    monkeypatch.setattr(
        tracking,
        "finish_wandb",
        lambda current, *, exit_code: events.append(("finish", exit_code)),
    )
    with pytest.raises(RuntimeError, match="seeding failed"):
        tracking.initialize_tracked_run(config, "/tmp/run", job_type="unit")
    assert events == [("wandb", "unit", None), ("seed", 29, 1), ("finish", 1)]
