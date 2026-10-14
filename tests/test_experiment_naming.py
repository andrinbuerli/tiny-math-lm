from __future__ import annotations

from hydra import compose, initialize_config_dir

from tinymath.tracking.naming import build_run_naming


def _exp11_config() -> dict:
    return {
        "seed": 18,
        "model": {"name": "10m", "context_length": 512},
        "experiment": {
            "number": 11,
            "name": "exp11_reasoning_500k_100k",
            "title": "Experiment 11 — Compact handwritten reasoning, 500k examples, 100k updates",
            "baseline": "restarted_100k_greedy_restart_2",
            "lineage": "restarted_100k_greedy_restart_2",
            "protocol": "reasoning_v1",
            "data_budget": 500000,
            "max_steps": 100000,
            "context_length": 512,
        },
        "tracking": {"project": "tinymath-rl"},
    }


def test_exp11_training_name_uses_actual_model_arm_and_seed():
    identity = build_run_naming(
        _exp11_config(), job_type="training", phase="main", model="10m", arm="grpo"
    )

    assert identity["project"] == "tinymath-rl"
    assert identity["group"] == "exp11-reasoning-500k-100k-seed18"
    assert identity["name"] == "exp11-reasoning-500k-100k-main-10m-grpo-seed18"
    assert identity["metadata"] == {
        "experiment_number": 11,
        "experiment_name": "exp11_reasoning_500k_100k",
        "experiment_title": "Experiment 11 — Compact handwritten reasoning, 500k examples, 100k updates",
        "phase": "main",
        "examples": 500000,
        "updates": 100000,
        "model": "10m",
        "arm": "grpo",
        "protocol": "reasoning_v1",
        "context": 512,
        "seed": 18,
        "baseline_identity": "restarted_100k_greedy_restart_2",
        "lineage": "restarted_100k_greedy_restart_2",
    }


def test_exp11_evaluation_identity_distinguishes_checkpoint():
    identity = build_run_naming(
        _exp11_config(),
        job_type="evaluation",
        model="10m",
        source_arm="continued_sft",
        checkpoint_selector="best",
        checkpoint_step=94000,
        checkpoint_hash="abcdeffedcba0123456789",
    )

    assert identity["name"] == (
        "exp11-reasoning-500k-100k-evaluation-10m-continued-sft-"
        "best-step94000-abcdeffedcba-seed18"
    )


def test_exp11_annotation_comparison_smoke_and_pilot_names_include_job_budgets():
    config = _exp11_config()
    assert build_run_naming(config, job_type="annotation")["name"] == (
        "exp11-reasoning-500k-100k-data-annotation-607000-rows-seed18"
    )
    assert build_run_naming(config, job_type="comparison")["name"] == (
        "exp11-reasoning-500k-100k-comparison-vs-exp10-seed18"
    )
    assert build_run_naming(
        config, job_type="smoke", model="1m", arm="sft", examples=2000, updates=20
    )["name"] == "exp11-reasoning-500k-100k-smoke-2000-examples-20-updates-1m-sft-seed18"
    assert build_run_naming(
        config, job_type="pilot", model="10m", arm="grpo", examples=2000, updates=200
    )["name"] == "exp11-reasoning-500k-100k-pilot-2000-examples-200-updates-10m-grpo-seed18"


def test_evaluation_name_requires_checkpoint_hash():
    import pytest

    with pytest.raises(ValueError, match="checkpoint_hash"):
        build_run_naming(
            _exp11_config(),
            job_type="evaluation",
            model="1m",
            source_arm="sft",
            checkpoint_selector="latest",
            checkpoint_step=100000,
        )


def test_legacy_experiment_names_are_returned_unchanged():
    identity = build_run_naming(
        {
            "seed": 17,
            "experiment": {"number": 10, "name": "restarted_100k_greedy_restart_2"},
            "tracking": {
                "project": "tinymath-rl",
                "group": "restarted_100k_greedy_restart_2",
                "name": "restarted_100k_greedy_restart_2-10m-seed17",
            },
        },
        job_type="training",
    )

    assert identity == {
        "project": "tinymath-rl",
        "group": "restarted_100k_greedy_restart_2",
        "name": "restarted_100k_greedy_restart_2-10m-seed17",
        "metadata": {},
    }


def test_exp11_hydra_preset_resolves_identity_bundle_seed_and_context():
    with initialize_config_dir(version_base=None, config_dir="/workspace/configs"):
        config = compose(config_name="config", overrides=["experiment=11_reasoning_500k_100k"])

    assert config.experiment.number == 11
    assert config.experiment.name == "exp11_reasoning_500k_100k"
    assert config.experiment.seed == 18
    assert config.seed == 18
    assert config.data_bundle == "data/processed/exp11_reasoning_500k_100k-seed18"
    assert config.experiment.source_bundle == "data/processed/d9f78d0665b3-main-seed18"
    assert config.model.context_length == 512
    assert config.data.context_length == 512
    assert config.rlvr.max_completion_tokens == 512
    assert config.experiment.max_steps == 100000
    identity = build_run_naming(config, job_type="training", phase="main", arm="sft")
    assert identity["group"] == "exp11-reasoning-500k-100k-seed18"
    assert identity["metadata"]["seed"] == 18


def test_annotation_hydra_config_uses_derived_bundle_and_canonical_wandb_identity():
    with initialize_config_dir(
        version_base=None, config_dir="/workspace/configs/annotation"
    ):
        config = compose(config_name="default")

    assert config.output_bundle == "/workspace/data/processed/exp11_reasoning_500k_100k-seed18"
    identity = build_run_naming(config, job_type="data-annotation", model="annotation")
    assert identity["name"] == "exp11-reasoning-500k-100k-data-annotation-607000-rows-seed18"
    assert identity["metadata"]["phase"] == "data-annotation"
    assert identity["metadata"]["protocol"] == "reasoning_v1"
