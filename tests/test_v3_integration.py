from __future__ import annotations

import csv
import json
import string
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from tinymath.checkpoint import compatibility_metadata, save_checkpoint, sha256_file
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.eval import evaluate as evaluation
from tinymath.eval.qualitative import build_qualitative_table
from tinymath.generation import RolloutConfig, Rollouts, generate_rollouts
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.repro import seed_everything
from tinymath.train import grpo_runner

GOOD = "<scp>\nstep:evaluate arithmetic operations in expression order\ncall:add(1,2) -> 3\n</scp>\nA: 3"
WRONG = GOOD[:-1] + "4"


class Scripted(torch.nn.Module):
    """A controlled trajectory source; no claim of on-policy sampling."""
    def __init__(self, tokenizer, texts, context_length):
        super().__init__()
        self.config = SimpleNamespace(context_length=context_length)
        self.texts = [tokenizer.encode(text, add_eos=True) for text in texts]
        self.tokenizer = tokenizer

    def forward_with_cache(self, ids, cache):
        position = cache or 0
        logits = torch.full((*ids.shape, self.tokenizer.vocab_size), -100., device=ids.device)
        for lane, script in enumerate(self.texts):
            token = script[position] if position < len(script) else self.tokenizer.eos_id
            logits[lane, -1, token] = 100.
        return logits, position + 1


def controlled_rollouts(model, prompt_ids, *, tokenizer, rollout, group_size=1, **kwargs):
    texts = [GOOD if lane % 3 == 0 else WRONG if lane % 3 == 1 else "malformed"
             for lane in range(len(prompt_ids) * group_size)]
    scripted = Scripted(tokenizer, texts, model.config.context_length)
    result = generate_rollouts(scripted, prompt_ids, group_size=group_size, tokenizer=tokenizer,
                               rollout=rollout, **kwargs)
    with torch.no_grad():
        old = grpo_runner.completion_log_probs(model, prompt_ids, result.completion_ids,
                                               temperature=kwargs["temperature"] or 1.)
    return Rollouts(result.completion_ids, result.completion_mask, old, result.entropy,
                    result.policy_mask, result.termination_reasons, result.call_events)


@pytest.fixture
def v3_setup(workspace_tmpdir):
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    tokenizer = CharacterTokenizer.from_texts([string.printable], protocol="reasoning_v3")
    tokenizer.save(bundle / "char_tokenizer.json")
    parent = {"question": "Calculate 1 + 2.", "answer": "3", "module": "arithmetic__add_or_sub",
              "is_sub_task": False}
    subtask = {**parent, "is_sub_task": True}
    for split in ("rlvr", "sft_validation", "test"):
        (bundle / f"{split}.jsonl").write_text(json.dumps(parent) + "\n" + json.dumps(subtask) + "\n")
    manifest = {"context_length": 256, "annotation": {"annotation_sha256": "fixture"}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    seed_everything(9, cpu_threads=1)
    model_config = GPTConfig(vocab_size=tokenizer.vocab_size, context_length=256, n_layers=1,
                            n_heads=1, d_model=16, d_ff=32)
    model = TinyMathGPT(model_config)
    checkpoint = workspace_tmpdir / "sft.pt"
    save_checkpoint(checkpoint, {
        "model": model.state_dict(), "model_config": model_config.__dict__,
        "tokenizer_sha256": tokenizer.sha256,
        "compatibility": compatibility_metadata(protocol="reasoning_v3", annotation_sha256="fixture",
            tokenizer_sha256=tokenizer.sha256, bundle_sha256=sha256_file(bundle / "manifest.json"), context_length=256),
    })
    config = SimpleNamespace(
        seed=9, rollout=RolloutConfig(executor="exact"),
        trainer=SimpleNamespace(bf16=False, weight_decay=0., grad_clip=1., save_every=1),
        rlvr=SimpleNamespace(updates=2, prompts_per_batch=1, group_size=3, temperature=1.,
                            max_completion_tokens=200, clip_epsilon=.2, kl_coefficient=.01, learning_rate=.001),
        tracking=SimpleNamespace(log_every=1),
        qualitative=SimpleNamespace(every_steps=20, examples_per_module=1),
    )
    return bundle, tokenizer, model, checkpoint, config


class Run:
    def __init__(self):
        self.logs = []
    def log(self, values):
        self.logs.append(values)


def test_controlled_update_parent_filter_and_exact_resume(v3_setup, workspace_tmpdir, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    bundle, _, initial_model, checkpoint, config = v3_setup
    monkeypatch.setattr(grpo_runner, "sample_group_rollouts", controlled_rollouts)
    run = Run()
    seed_everything(20, cpu_threads=1)
    uninterrupted = grpo_runner.train_grpo(config=config, bundle_dir=bundle, init_checkpoint=checkpoint,
                                          run_dir=workspace_tmpdir / "whole", run=run)
    assert uninterrupted["prompt_examples"] == 1
    assert uninterrupted["final_reward"] == pytest.approx(1.1 / 3)
    whole = torch.load(uninterrupted["checkpoint"], weights_only=False)
    assert any(not torch.equal(value, whole["model"][name]) for name, value in initial_model.state_dict().items())
    assert whole["policy_tokens_seen"] + whole["executor_tokens_seen"] == whole["tokens_seen"]
    assert whole["executor_tokens_seen"] == 4
    assert run.logs[-1]["grpo/reward/format_only_rate"] == pytest.approx(1 / 3)
    original_save = grpo_runner.save_checkpoint
    def interrupt(path, payload):
        original_save(path, payload)
        if path.name == "latest_inference.pt" and payload["update"] == 1:
            raise KeyboardInterrupt("controlled checkpoint interruption")
    monkeypatch.setattr(grpo_runner, "save_checkpoint", interrupt)
    seed_everything(20, cpu_threads=1)
    with pytest.raises(KeyboardInterrupt):
        grpo_runner.train_grpo(config=config, bundle_dir=bundle, init_checkpoint=checkpoint,
                              run_dir=workspace_tmpdir / "resumed", run=Run())
    monkeypatch.setattr(grpo_runner, "save_checkpoint", original_save)
    resumed = grpo_runner.train_grpo(config=config, bundle_dir=bundle, init_checkpoint=checkpoint,
                                    run_dir=workspace_tmpdir / "resumed", run=Run())
    payload = torch.load(resumed["checkpoint"], weights_only=False)
    assert all(torch.equal(value, payload["model"][name]) for name, value in whole["model"].items())
    completed = grpo_runner.train_grpo(config=config, bundle_dir=bundle, init_checkpoint=checkpoint,
                                      run_dir=workspace_tmpdir / "resumed", run=Run())
    assert completed["final_reward"] == resumed["final_reward"]
    config.rollout = RolloutConfig(executor="none")
    with pytest.raises(ValueError, match="rollout/reward"):
        grpo_runner.train_grpo(config=config, bundle_dir=bundle, init_checkpoint=checkpoint,
                              run_dir=workspace_tmpdir / "resumed", run=Run())


def test_evaluation_and_qualitative_share_scoring_and_parent_population(v3_setup, workspace_tmpdir, monkeypatch):
    bundle, tokenizer, model, checkpoint, _ = v3_setup
    monkeypatch.setattr(evaluation, "generate_rollouts", controlled_rollouts)
    config = RolloutConfig(executor="exact")
    kwargs = {"checkpoint_path": checkpoint, "bundle_dir": bundle, "output_dir": workspace_tmpdir / "eval",
              "split_names": ["test"], "max_new_tokens": 200, "rollout": config}
    summary = evaluation.evaluate_checkpoint(**kwargs)
    result = summary["splits"]["test"]
    assert result["examples"] == 1
    assert result["accuracy"] == result["full_success_rate"] == result["mean_reward"] == 1
    assert result["executor_tokens"] == 1
    assert summary["assisted"]
    assert evaluation.evaluate_checkpoint(**kwargs) == summary
    with pytest.raises(ValueError, match="progress"):
        evaluation.evaluate_checkpoint(**{**kwargs, "rollout": RolloutConfig()})
    from tinymath.eval import qualitative
    monkeypatch.setattr(qualitative, "generate_rollouts", controlled_rollouts)
    parent = {"question": "Calculate 1 + 2.", "answer": "3", "module": "arithmetic__add_or_sub"}
    table, count, accuracy = build_qualitative_table(model=model, tokenizer=tokenizer,
        split_rows={"test": [parent, {**parent, "is_sub_task": True}]}, training_arm="sft", seed=9,
        max_new_tokens=200, rollout=config)
    assert count == 1 and accuracy["test"] == 1
    record = dict(zip(table.columns, table.data[0], strict=True))
    assert record["reward"] == 1 and record["full_success"] and record["assisted"]
    with (workspace_tmpdir / "eval" / "test_predictions.csv").open() as handle:
        assert next(csv.DictReader(handle))["reward"] == "1.0"


def test_v3_hydra_configuration():
    from pathlib import Path
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="reasoning_v3", overrides=["rollout.executor=none"])
    assert config.model.context_length == config.rlvr.max_completion_tokens == config.eval.max_new_tokens == 512
    assert RolloutConfig.from_config(config.rollout).executor == "none"
