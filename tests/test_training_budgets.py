from __future__ import annotations

from collections import Counter

from hydra import compose, initialize_config_dir

from tinymath.train.budgets import budget_rows, isolated_training_run_dir


def test_budget_rows_are_deterministic_balanced_and_keep_source_order():
    rows = [
        {"module": f"module_{index % 5}", "question": str(index)}
        for index in range(100)
    ]
    first = budget_rows(rows, examples=12, seed=18)
    second = budget_rows(rows, examples=12, seed=18)

    assert first == second
    assert [int(row["question"]) for row in first] == sorted(
        int(row["question"]) for row in first
    )
    assert Counter(row["module"] for row in first) == {
        "module_0": 3,
        "module_1": 3,
        "module_2": 2,
        "module_3": 2,
        "module_4": 2,
    }


def test_budget_rows_redistribute_quotas_when_one_module_is_small():
    rows = [{"module": "small", "question": "only"}]
    rows.extend({"module": "large", "question": str(index)} for index in range(9))
    selected = budget_rows(rows, examples=6, seed=5)

    assert len(selected) == 6
    assert Counter(row["module"] for row in selected) == {"large": 5, "small": 1}


def test_smoke_and_pilot_run_paths_are_isolated_from_main():
    with initialize_config_dir(version_base=None, config_dir="/workspace/configs"):
        main = compose(
            config_name="config", overrides=["experiment=11_reasoning_500k_100k"]
        )
        pilot = compose(
            config_name="config",
            overrides=[
                "experiment=11_reasoning_500k_100k",
                "experiment.phase=pilot",
                "experiment.data_budget=2000",
                "experiment.max_steps=10",
                "experiment.updates=10",
            ],
        )

    assert isolated_training_run_dir(main, model="1m", arm="sft") == (
        "/workspace/runs/exp11_reasoning_500k_100k/1m/sft/seed18"
    )
    assert isolated_training_run_dir(pilot, model="1m", arm="sft") == (
        "/workspace/runs/exp11_reasoning_500k_100k/1m/sft/pilot-2000-examples-10-updates/seed18"
    )
