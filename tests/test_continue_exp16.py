from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf

SPEC = importlib.util.spec_from_file_location(
    "continue_exp16", Path(__file__).resolve().parents[1] / "scripts" / "continue_exp16.py"
)
assert SPEC and SPEC.loader
CONTINUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTINUE)


def test_continuation_child_command_uses_hydra_config_group(monkeypatch):
    observed = {}
    monkeypatch.setattr(CONTINUE.shutil, "which", lambda _name: "/workspace/.tools/uv")
    monkeypatch.setattr(
        CONTINUE.subprocess,
        "run",
        lambda command, **kwargs: observed.update(command=command, **kwargs),
    )

    CONTINUE._run(
        OmegaConf.create(
            {"experiment": {"config_group": "16_qwen3_base_reasoning_v3"}}
        ),
        "train_rlvr",
        "experiment.phase=main",
        "rlvr.updates=10000",
    )

    assert observed["command"][5:] == [
        "experiment=16_qwen3_base_reasoning_v3",
        "experiment.phase=main",
        "rlvr.updates=10000",
    ]
    assert observed["cwd"] == "/workspace"
    assert observed["check"] is True


def test_continuation_evaluations_use_memory_bounded_batch(monkeypatch):
    observed = {}
    monkeypatch.setattr(
        CONTINUE,
        "_run",
        lambda _config, script, *overrides: observed.update(
            script=script, overrides=overrides
        ),
    )

    CONTINUE._evaluate(
        OmegaConf.create({"experiment": {"config_group": "16_qwen3_base_reasoning_v3"}}),
        "sft_evaluation",
        "runs/model/sft/best",
    )

    assert observed["script"] == "evaluate"
    assert observed["overrides"] == (
        "experiment.phase=sft_evaluation",
        "checkpoint=runs/model/sft/best",
        "eval.batch_size=4",
    )


def test_completed_stage_requires_finished_run_and_artifacts(tmp_path: Path):
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"run_status": "running"}), encoding="utf-8"
    )
    assert not CONTINUE._stage_finished(tmp_path, "result.json")

    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"run_status": "finished"}), encoding="utf-8"
    )
    assert CONTINUE._stage_finished(tmp_path, "result.json")
    assert not CONTINUE._stage_finished(tmp_path, "missing.json")
