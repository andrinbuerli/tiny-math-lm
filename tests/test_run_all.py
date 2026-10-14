from __future__ import annotations

import importlib.util
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
