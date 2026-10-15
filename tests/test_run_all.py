from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

spec = importlib.util.spec_from_file_location("run_all", Path(__file__).resolve().parents[1] / "scripts" / "run_all.py")
assert spec is not None and spec.loader is not None
run_all = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_all)


def test_orchestrator_emits_hydra_valid_mapping_overrides_and_pilot_budgets():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="config",
            overrides=[
                "experiment=11_reasoning_500k_100k",
                "experiment.phase=pilot",
                "experiment.data_budget=2000",
                "experiment.validation_examples=100",
                "experiment.max_steps=10",
                "experiment.updates=10",
                "tracking.mode=offline",
                "reporting.output_dir=/workspace/reports/custom",
                "reporting.annotation_bundle_dir=/workspace/data/custom",
            ],
        )
    values = run_all._overrides(
        config,
        model="1m",
        trainer="sft",
        experiment_choice="11_reasoning_500k_100k",
    )

    assert "experiment.phase=\"pilot\"" in values
    assert "experiment.data_budget=2000" in values
    assert "experiment.max_steps=10" in values
    assert "runtime.parallel_model_vram_gb={1m:1,10m:3,100m:8}" in values
    assert "tracking.mode=\"offline\"" in values
    assert 'reporting.output_dir="/workspace/reports/custom"' in values
    assert 'reporting.annotation_bundle_dir="/workspace/data/custom"' in values


def test_orchestrator_propagates_reasoning_v3_rollout_to_child_runs():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="reasoning_v3",
            overrides=["experiment=13_reasoning_v3_arithmetic_subtasks"],
        )
    values = run_all._overrides(
        config,
        model="10m",
        trainer="grpo",
        experiment_choice="13_reasoning_v3_arithmetic_subtasks",
    )
    assert 'rollout.executor="exact"' in values
    assert "rollout.token_accounting=\"transcript_includes_executor_v1\"" in values


def test_orchestrator_uses_current_exact_v3_rollout_after_resumed_sft():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="config",
            overrides=[
                "experiment=13_reasoning_v3_arithmetic_subtasks",
                "rollout.rollout_version=2",
                "rollout.scorer_version=2",
                "rollout.delegation_policy=proper_subexpression_v1",
            ],
        )

    downstream = run_all._downstream_config(config)
    assert config.rollout.rollout_version == 2
    assert downstream.rollout.executor == "exact"
    assert downstream.rollout.rollout_version == 3
    assert downstream.rollout.scorer_version == 3
    assert downstream.rollout.delegation_policy == "compound_task_v1"


def test_orchestrator_preserves_literal_backslash_in_saved_experiment_title():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    title = "Experiment 13 \\u2014 Reasoning-v3 arithmetic subtasks"
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(
            config_name="reasoning_v3",
            overrides=[
                "experiment=13_reasoning_v3_arithmetic_subtasks",
                f"experiment.title='{title}'",
            ],
        )
        child_overrides = run_all._overrides(
            config,
            model="100m",
            trainer="sft",
            experiment_choice="13_reasoning_v3_arithmetic_subtasks",
        )
        child_config = compose(config_name="reasoning_v3", overrides=child_overrides)

    assert config.experiment.title == title
    assert child_config.experiment.title == title


def test_orchestrator_names_each_fresh_arm_run_and_wandb_group():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(
            config_name="reasoning_v3",
            overrides=[
                "experiment=13_reasoning_v3_arithmetic_subtasks",
                "experiment.name=exp13-fresh-test",
                "tracking.group=exp13-fresh-test",
            ],
        )
    values = run_all._overrides(
        config,
        model="10m",
        trainer="sft",
        experiment_choice="13_reasoning_v3_arithmetic_subtasks",
        run_arm="grpo",
        run_label="training",
    )

    assert 'tracking.name="exp13-fresh-test-10m-grpo-training-seed18"' in values
    assert 'tracking.group="exp13-fresh-test-10m-grpo"' in values
    assert 'tracking.tags=["exp13-fresh-test", "10m", "grpo"]' in values
    assert 'experiment.name="exp13-fresh-test"' in values


def test_gpu_admission_waits_for_telemetry_before_launch(monkeypatch):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(
            config_name="reasoning_v3",
            overrides=["experiment=13_reasoning_v3_arithmetic_subtasks"],
        )

    readings = iter([subprocess.CalledProcessError(1, "nvidia-smi"), (25, 20 * 1024)])

    def read_gpu():
        reading = next(readings)
        if isinstance(reading, Exception):
            raise reading
        return reading

    monkeypatch.setattr(run_all, "_gpu_load", read_gpu)
    monkeypatch.setattr(run_all.time, "sleep", lambda _seconds: None)
    run_all._wait_for_gpu_capacity(config, "100m")


def test_orchestrator_can_defer_comparison_until_all_model_pipelines_finish(monkeypatch):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(
            config_name="reasoning_v3",
            overrides=["experiment=13_reasoning_v3_arithmetic_subtasks"],
        )

    calls = []
    monkeypatch.setattr(run_all, "_run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setenv("TINYMATH_SKIP_COMPARISON", "1")
    run_all._run_comparison(config)
    assert calls == []

    monkeypatch.delenv("TINYMATH_SKIP_COMPARISON")
    run_all._run_comparison(config)
    assert len(calls) == 1
    assert calls[0][0] == "plot.py"
