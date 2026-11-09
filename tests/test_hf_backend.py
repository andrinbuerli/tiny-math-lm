import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from omegaconf import OmegaConf

from tiny_math_lm.data.hf_reasoning import score_native_reasoning
from tiny_math_lm.generation import RolloutConfig
from tiny_math_lm.model.hf_qwen import completion_loss
from tiny_math_lm.runtime_gpu import gpu_capacity_available
from tiny_math_lm.train import hf_sft
from tiny_math_lm.train.grpo import group_relative_advantages, grpo_objective
from tiny_math_lm.train.hf_generation import (
    NativeRollout,
    generate_native_rollout,
    generate_native_rollout_batch,
    generate_native_rollout_group,
)
from tiny_math_lm.train.hf_grpo import PromptRollouts, chunked_grpo_backward, native_policy_statistics


def test_sft_resume_truncates_metrics_after_durable_checkpoint(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    fields = ["optimizer_step", "train/completion_loss"]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"optimizer_step": 20449, "train/completion_loss": 0.01},
                {"optimizer_step": 20450, "train/completion_loss": 0.009},
                {"optimizer_step": 20451, "train/completion_loss": 0.008},
            ]
        )

    removed = hf_sft._truncate_metrics_after_checkpoint(
        metrics,
        fields=fields,
        optimizer_step=20450,
        pending_validation_metrics=None,
    )

    assert removed == 1
    with metrics.open(newline="", encoding="utf-8") as handle:
        assert [int(row["optimizer_step"]) for row in csv.DictReader(handle)] == [20449, 20450]


def test_sft_resume_repeats_pending_validation_row(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    fields = ["optimizer_step", "train/completion_loss"]
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"optimizer_step": 499, "train/completion_loss": 0.02},
                {"optimizer_step": 500, "train/completion_loss": 0.01},
            ]
        )

    removed = hf_sft._truncate_metrics_after_checkpoint(
        metrics,
        fields=fields,
        optimizer_step=500,
        pending_validation_metrics={"optimizer_step": 500},
    )

    assert removed == 1
    with metrics.open(newline="", encoding="utf-8") as handle:
        assert [int(row["optimizer_step"]) for row in csv.DictReader(handle)] == [499]


def test_sft_resume_wandb_offset_clears_history_ahead_of_checkpoint(monkeypatch):
    fake_wandb = SimpleNamespace(
        Api=lambda: SimpleNamespace(
            run=lambda _path: SimpleNamespace(
                summary={"_step": 20961},
                scan_history=lambda **_kwargs: [
                    {"_step": 20960, "optimizer_step": 20461}
                ],
            )
        )
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run = SimpleNamespace(entity="entity", project="project", id="run-id")

    offset = hf_sft._resume_wandb_step_offset(
        run, optimizer_step=20450, local_last_step=20461
    )

    assert offset == 512


def test_pilot_subset_always_contains_longest_native_token_sequence():
    import numpy as np

    class Split:
        offsets = np.array([0, 2, 8, 12, 30, 34], dtype=np.uint64)

        def __len__(self):
            return 5

    first = hf_sft._pilot_indices(Split(), row_cap=3, seed=18)
    second = hf_sft._pilot_indices(Split(), row_cap=3, seed=18)
    assert first == second
    assert 3 in first
    assert len(first) == 3


def test_compute_match_plans_first_optimizer_boundary_at_or_over_flop_target():
    import numpy as np

    class Split:
        offsets = np.array([0, 3, 7, 12, 18], dtype=np.uint64)

    planned = hf_sft._plan_compute_matched_updates(
        Split(),
        [0, 1, 2, 3],
        seed=18,
        parameters=10,
        target_flops=6 * 10 * 5,
        batch_size=1,
        accumulation=2,
    )
    repeated = hf_sft._plan_compute_matched_updates(
        Split(),
        [0, 1, 2, 3],
        seed=18,
        parameters=10,
        target_flops=6 * 10 * 5,
        batch_size=1,
        accumulation=2,
    )
    updates, tokens, epochs = planned
    assert planned == repeated
    assert updates == 1
    assert tokens >= 5
    assert epochs == 1


def test_cosine_schedule_reaches_zero_on_the_preplanned_final_update():
    assert hf_sft._cosine_lr(0, warmup_steps=3, total_steps=10, base_lr=1.0) == 1 / 3
    assert hf_sft._cosine_lr(9, warmup_steps=3, total_steps=10, base_lr=1.0) == 0.0


def test_periodic_greedy_accuracy_is_logged_with_validation_loss(monkeypatch, tmp_path):
    config = OmegaConf.create(
        {
            "eval": {
                "batch_size": 2,
                "greedy_accuracy_enabled": True,
                "greedy_accuracy_every_steps": 10,
                "greedy_accuracy_max_examples": 8,
                "greedy_accuracy_batch_size": 4,
                "greedy_accuracy_max_new_tokens": 32,
            },
            "seed": 18,
            "model": {"context_length": 512, "revision": "revision"},
            "rollout": {"executor": "exact"},
            "experiment": {"phase": "main"},
        }
    )
    monkeypatch.setattr(hf_sft, "_evaluate_loss", lambda *_args, **_kwargs: 0.75)
    calls = []

    def fake_accuracy(**kwargs):
        calls.append(kwargs)
        return 0.375, 8

    monkeypatch.setattr(hf_sft, "evaluate_native_greedy_exact_match", fake_accuracy)

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, values, *, step):
            self.logged.append((values, step))

    run = FakeRun()
    values = {"optimizer_step": 20}
    validation_loss, best_loss = hf_sft._validate_and_select_best(
        model=object(),
        validation_split=object(),
        config=config,
        device=torch.device("cpu"),
        pad_id=0,
        validation_cap=None,
        values=values,
        best_val=0.5,
        best_dir=tmp_path / "best",
        cache_manifest={},
        cache_dir=tmp_path,
        validation_rows=[{"question": "1 + 1", "answer": "2"}],
        tokenizer=object(),
        init_checkpoint=None,
        trainer_name="sft",
        parameters=1,
        run=run,
        wandb_step=120,
    )

    assert (validation_loss, best_loss) == (0.75, 0.5)
    assert calls[0]["max_examples"] == 8
    assert values["validation/greedy_accuracy"] == 0.375
    assert values["validation/greedy_accuracy_examples"] == 8
    assert len(run.logged) == 1
    logged, step = run.logged[0]
    assert logged == {
        "validation/completion_loss": 0.75,
        "validation/greedy_accuracy": 0.375,
        "validation/greedy_accuracy_examples": 8,
    }
    assert step == 120


def test_metrics_schema_upgrade_preserves_rows_and_adds_validation_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "optimizer_step,validation/completion_loss\n10,0.5\n20,0.4\n",
        encoding="utf-8",
    )
    fields = [
        "optimizer_step",
        "validation/completion_loss",
        "validation/greedy_accuracy",
    ]

    hf_sft._upgrade_metrics_schema(path, fields)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "optimizer_step": "10",
            "validation/completion_loss": "0.5",
            "validation/greedy_accuracy": "",
        },
        {
            "optimizer_step": "20",
            "validation/completion_loss": "0.4",
            "validation/greedy_accuracy": "",
        },
    ]


def test_resume_wandb_step_offset_uses_existing_run_history(monkeypatch):
    import sys
    from types import SimpleNamespace

    class RemoteRun:
        summary = {"_step": 12973}

        def scan_history(self, *, keys, min_step, page_size):
            assert keys == ["_step", "optimizer_step"]
            assert min_step == 12963
            assert page_size == 1000
            return iter(
                [
                    {"_step": 12971, "optimizer_step": 12472},
                    {"_step": 12973, "optimizer_step": 12474},
                ]
            )

    class FakeApi:
        def run(self, path):
            assert path == "entity/project/run-id"
            return RemoteRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=FakeApi))
    run = SimpleNamespace(entity="entity", project="project", id="run-id")

    offset = hf_sft._resume_wandb_step_offset(
        run, optimizer_step=12500, local_last_step=12499
    )

    assert offset == 499


class TinyLogitModel(torch.nn.Module):
    def __init__(self, vocab_size=7):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, attention_mask, use_cache=False):
        batch, length = input_ids.shape
        return type("Output", (), {"logits": self.logits.expand(batch, length, -1)})()


def test_completion_loss_counts_only_completion_targets_including_native_eos():
    model = TinyLogitModel()
    input_ids = torch.tensor([[1, 2, 3, 6, 6]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, 4, -100]])

    loss_sum, token_count = completion_loss(model, input_ids, attention_mask, labels)

    assert token_count == 1
    loss_sum.backward()
    assert model.logits.grad is not None


def test_completion_loss_preserves_eos_when_pad_uses_same_token_id():
    model = TinyLogitModel()
    eos_id = 6
    input_ids = torch.tensor([[1, 2, 3, eos_id, eos_id]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, eos_id, -100]])

    _loss_sum, token_count = completion_loss(model, input_ids, attention_mask, labels)

    assert token_count == 1


def test_qwen_training_admission_requires_utilization_and_vram_headroom():
    config = OmegaConf.create(
        {
            "runtime": {
                "parallel_gpu_max_utilization": 75,
                "parallel_gpu_min_free_vram_gb": 2,
                "parallel_model_vram_gb": {"qwen3_0_6b_base": 10},
            }
        }
    )
    assert gpu_capacity_available(config, "qwen3_0_6b_base", 60, 12 * 1024)
    assert not gpu_capacity_available(config, "qwen3_0_6b_base", 76, 16 * 1024)
    assert not gpu_capacity_available(config, "qwen3_0_6b_base", 60, 12 * 1024 - 1)


class TinyCachedSplit:
    offsets: ClassVar[tuple[int, ...]] = (0, 4, 8, 12, 16)

    def __init__(self, _cache_dir, split):
        self.split = split

    def __len__(self):
        return 4

    def batch(self, indices, *, pad_id, device):
        del pad_id
        input_ids = torch.tensor([[1, 2, 3, 6] for _ in indices], device=device)
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, 3, 6] for _ in indices], device=device)
        return input_ids, attention, labels


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(7))

    def forward(self, input_ids, attention_mask, use_cache=False):
        batch, length = input_ids.shape
        return type("Output", (), {"logits": self.logits.expand(batch, length, -1)})()

    def save_pretrained(self, path, safe_serialization=True):
        del safe_serialization
        path.mkdir(parents=True, exist_ok=True)
        (path / "tiny-model.json").write_text("{}", encoding="utf-8")
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"tiny fixture weights")


class TinyTokenizer:
    def save_pretrained(self, path):
        (path / "tiny-tokenizer.json").write_text("{}", encoding="utf-8")


def test_cpu_fixture_resumes_deterministically_from_optimizer_boundary(monkeypatch, tmp_path):
    config = OmegaConf.create(
        {
            "seed": 18,
            "experiment": {"phase": "pilot", "pilot_train_rows": 4, "pilot_validation_rows": 2},
            "model": {
                "revision": "revision",
                "context_length": 512,
                "gradient_checkpointing": True,
            },
            "trainer": {
                "name": "sft",
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "batch_size": 1,
                "gradient_accumulation": 2,
                "max_epochs": 1,
                "max_steps": 2,
                "eval_every": 1,
                "save_every": 1,
                "warmup_ratio": 0.0,
                "grad_clip": 1.0,
                "early_stopping_patience_epochs": 1,
            },
            "eval": {"batch_size": 1},
        }
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = {
        "tokenizer_identity": {"pad_token_id": 6},
        "tokenizer_sha256": "tokenizer-hash",
        "model_config_sha256": "config-hash",
    }
    monkeypatch.setattr(hf_sft, "_find_cache", lambda _config: (cache_dir, manifest))
    monkeypatch.setattr(hf_sft, "HFTokenSplit", TinyCachedSplit)
    monkeypatch.setattr(hf_sft, "load_model", lambda *_args, **_kwargs: TinyTrainModel())
    monkeypatch.setattr(hf_sft, "load_tokenizer", lambda _config: TinyTokenizer())
    checkpoint_dir = tmp_path / "run"

    def stop_after_first_boundary(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        if path.name == "latest.pt" and payload["step"] == 1:
            raise RuntimeError("simulated interruption after complete optimizer update")

    monkeypatch.setattr(hf_sft, "save_checkpoint", stop_after_first_boundary)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        hf_sft.train_hf_sft(
            config=config,
            run_dir=checkpoint_dir,
            run=None,
            device_override=torch.device("cpu"),
        )
    interrupted = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert interrupted["step"] == 1
    assert interrupted["cursor"] == 2
    assert interrupted["pending_validation_metrics"]["optimizer_step"] == 1

    def save_checkpoint(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    monkeypatch.setattr(hf_sft, "save_checkpoint", save_checkpoint)
    result = hf_sft.train_hf_sft(
        config=config,
        run_dir=checkpoint_dir,
        run=None,
        device_override=torch.device("cpu"),
    )
    assert result["step"] == 2
    assert result["examples_seen"] == 4
    assert result["tokens_seen"] > 0
    assert result["planned_epochs"] is None
    assert (checkpoint_dir / "best" / "experiment_identity.json").is_file()
    assert (checkpoint_dir / "training_summary.json").is_file()
    assert (checkpoint_dir / "final" / "experiment_identity.json").is_file()
    completed = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert completed["step"] == 2
    assert completed["epoch"] == 1
    with (checkpoint_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert metric_rows[0]["optimizer_step"] == "1"
    assert metric_rows[0]["validation/completion_loss"]


def test_continued_sft_stops_at_planned_compute_boundary_and_selects_final(
    monkeypatch, workspace_tmpdir
):
    config = OmegaConf.create(
        {
            "seed": 18,
            "experiment": {"phase": "continued_sft"},
            "model": {
                "revision": "revision",
                "context_length": 512,
                "gradient_checkpointing": True,
            },
            "trainer": {
                "name": "continued_sft",
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "batch_size": 1,
                "gradient_accumulation": 2,
                "max_epochs": 1,
                "max_steps": 2,
                "eval_every": 500,
                "save_every": 1,
                "warmup_ratio": 0.03,
                "grad_clip": 1.0,
                "early_stopping_patience_epochs": 1,
                "compute_budget_flops": 336,
            },
            "eval": {"batch_size": 1},
        }
    )
    cache_dir = workspace_tmpdir / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = {
        "tokenizer_identity": {"pad_token_id": 6},
        "tokenizer_sha256": "tokenizer-hash",
        "model_config_sha256": "config-hash",
    }
    monkeypatch.setattr(hf_sft, "_find_cache", lambda _config: (cache_dir, manifest))
    monkeypatch.setattr(hf_sft, "HFTokenSplit", TinyCachedSplit)
    monkeypatch.setattr(hf_sft, "load_model", lambda *_args, **_kwargs: TinyTrainModel())
    monkeypatch.setattr(hf_sft, "load_tokenizer", lambda _config: TinyTokenizer())
    init_checkpoint = workspace_tmpdir / "sft" / "best"
    result = hf_sft.train_hf_sft(
        config=config,
        run_dir=workspace_tmpdir / "continued_sft",
        run=None,
        init_checkpoint=init_checkpoint,
        device_override=torch.device("cpu"),
    )
    assert result["step"] == 1
    assert result["planned_optimizer_steps"] == 1
    assert result["additional_estimated_flops"] == 336
    assert result["compute_matching_error"] == 0.0
    assert result["checkpoint"] == str(workspace_tmpdir / "continued_sft" / "final")
    latest_state = torch.load(
        workspace_tmpdir / "continued_sft" / "latest.pt", map_location="cpu", weights_only=False
    )
    assert latest_state["step"] == 1
    final_identity = json.loads(
        (workspace_tmpdir / "continued_sft" / "final" / "experiment_identity.json").read_text()
    )
    assert final_identity["source_sft_checkpoint"] == str(init_checkpoint)


class MultiTokenTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 8

    def __init__(self, call_text):
        self.pieces = {
            5: call_text,
            6: "\nstep:+(3,0)\n</think>\nA: 3",
            7: "3",
        }

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces.get(token, "") for token in token_ids if token != 2)

    def encode(self, text, **_kwargs):
        return [7] if text == "3" else []


class CachedGenerationModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        del attention_mask, use_cache
        prefix = () if past_key_values is None else past_key_values
        new_ids = tuple(int(token) for token in input_ids.reshape(-1).tolist())
        sequence = (*prefix, *new_ids)
        next_id = 5 if sequence[-1] == 1 else 6 if sequence[-1] == 7 else 2
        logits = (
            torch.full((1, input_ids.shape[1], self.vocab_size), -20.0, device=input_ids.device)
            + self.anchor * 0
        )
        logits[:, -1, next_id] = 20.0
        return type("Output", (), {"logits": logits, "past_key_values": sequence})()


def test_native_rollout_handles_multichar_tokens_and_masks_executor_observations():
    call_text = (
        "<think>\nstep:evaluate arithmetic operations in expression order\ncall:add(1,2) -> "
    )
    tokenizer = MultiTokenTokenizer(call_text)
    row = {"question": "Calculate (1+2)+0.", "answer": "3"}
    rollout = generate_native_rollout(
        CachedGenerationModel(),
        tokenizer,
        [1],
        row,
        context_length=128,
        max_completion_tokens=32,
        temperature=1.0,
        rollout=RolloutConfig(executor="exact"),
        greedy=True,
    )
    assert rollout.termination_reason == "eos"
    assert rollout.token_ids == [5, 7, 6, 2]
    assert rollout.policy_mask == [True, False, True, True]
    assert rollout.executor_tokens == 1
    assert rollout.call_events[0]["result"] == "3"
    assert rollout.old_log_probs[1] == rollout.entropy[1] == 0.0
    assert all(math.isfinite(value) for value in rollout.entropy)
    text = tokenizer.decode(rollout.token_ids[:-1])
    assert score_native_reasoning(text, {**row, "scratchpad": text.split("\n", 1)[1]}).format_valid


class BatchRolloutTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, pieces=None):
        self.pieces = pieces or {1: "x", 2: "y", 3: "z"}

    def __len__(self):
        return 4

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces.get(int(token), "") for token in token_ids)


class ScriptedBatchGenerator(torch.nn.Module):
    def __init__(self, sequences, scores):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.sequences = sequences
        self.scores = scores

    def generate(self, input_ids, **_kwargs):
        torch.rand(())
        return type(
            "GenerationOutput",
            (),
            {
                "sequences": torch.cat((input_ids, self.sequences.to(input_ids.device)), dim=1),
                "scores": tuple(score.to(input_ids.device) for score in self.scores),
            },
        )()


def test_native_rollout_group_batches_no_call_sampling_and_keeps_scores_aligned():
    model = ScriptedBatchGenerator(
        torch.tensor([[1, 0], [2, 3]]),
        (
            torch.tensor([[0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 2.0, -1.0]]),
            torch.tensor([[2.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 2.0]]),
        ),
    )
    model.train()
    rollouts = generate_native_rollout_group(
        model,
        BatchRolloutTokenizer(),
        [3],
        {"question": "1 + 1", "answer": "2"},
        group_size=2,
        context_length=8,
        max_completion_tokens=2,
        temperature=1.0,
    )
    assert model.training
    assert [rollout.token_ids for rollout in rollouts] == [[1, 0], [2, 3]]
    assert [rollout.termination_reason for rollout in rollouts] == ["eos", "budget_exceeded"]
    assert [rollout.policy_tokens for rollout in rollouts] == [2, 2]
    assert rollouts[0].old_log_probs[0] == pytest.approx(
        torch.log_softmax(model.scores[0][0], dim=-1)[1].item()
    )
    assert all(math.isfinite(value) for rollout in rollouts for value in rollout.entropy)


def test_native_rollout_group_replays_executor_groups_from_saved_rng(monkeypatch):
    from tiny_math_lm.train import hf_generation

    model = ScriptedBatchGenerator(
        torch.tensor([[1], [1]]),
        (torch.zeros((2, 4)),),
    )
    tokenizer = BatchRolloutTokenizer({1: "<think>\ncall:add(1,2) -> ", 0: ""})
    expected_state = torch.get_rng_state().clone()
    observed_states = []

    def replay(*_args, **_kwargs):
        observed_states.append(torch.equal(torch.get_rng_state(), expected_state))
        return NativeRollout([1], [True], [-0.1], [0.5], "eos", ())

    monkeypatch.setattr(hf_generation, "generate_native_rollout", replay)
    rollouts = generate_native_rollout_group(
        model,
        tokenizer,
        [3],
        {"question": "(1+2)+0", "answer": "3"},
        group_size=2,
        context_length=8,
        max_completion_tokens=1,
        temperature=1.0,
    )
    assert len(rollouts) == 2
    assert observed_states == [True, True]


class SelectableFakeCache:
    def __init__(self, markers, steps):
        self.markers = markers
        self.steps = steps

    def batch_select_indices(self, indices):
        self.markers = self.markers.index_select(0, indices)
        self.steps = self.steps.index_select(0, indices)


class IncrementalBatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values=None, use_cache=True
    ):
        del position_ids, use_cache
        if past_key_values is None:
            markers = input_ids[:, -1]
            steps = torch.zeros_like(markers)
        else:
            markers = past_key_values.markers
            steps = past_key_values.steps + 1
        next_ids = torch.where(
            markers == 4,
            torch.where(steps == 0, 1, 0),
            torch.where(
                markers == 5,
                torch.where(steps == 0, 2, torch.where(steps == 1, 3, 0)),
                3,
            ),
        )
        logits = (
            torch.full(
                (input_ids.shape[0], input_ids.shape[1], 8),
                -20.0,
                device=input_ids.device,
            )
            + self.anchor * 0
        )
        logits[:, -1, :].scatter_(1, next_ids[:, None], 20.0)
        return type(
            "Output",
            (),
            {
                "logits": logits,
                "past_key_values": SelectableFakeCache(markers, steps),
            },
        )()


def test_native_rollout_batch_compacts_finished_rows_and_keeps_context_caps():
    model = IncrementalBatchModel()
    model.train()
    rollouts = generate_native_rollout_batch(
        model,
        BatchRolloutTokenizer({0: "", 1: "x", 2: "y", 3: "z", 4: "w"}),
        [
            ([4], {"question": "a", "answer": "x"}),
            ([5], {"question": "b", "answer": "yz"}),
            ([6, 6, 6], {"question": "c", "answer": "w"}),
        ],
        context_length=4,
        max_completion_tokens=3,
        temperature=1.0,
        rollout=RolloutConfig(executor="exact"),
    )
    assert model.training
    assert [rollout.token_ids for rollout in rollouts] == [[1, 0], [2, 3, 0], [3]]
    assert [rollout.termination_reason for rollout in rollouts] == ["eos", "eos", "budget_exceeded"]
    assert all(rollout.policy_tokens == len(rollout.token_ids) for rollout in rollouts)
    assert all(math.isfinite(value) for rollout in rollouts for value in rollout.entropy)


def test_native_rollout_batch_replays_all_rows_when_any_executor_call_appears(monkeypatch):
    from tiny_math_lm.train import hf_generation

    model = IncrementalBatchModel()
    tokenizer = BatchRolloutTokenizer(
        {0: "", 1: "<think>\ncall:add(1,2) -> ", 2: "y", 3: "z", 4: "w"}
    )
    expected_state = torch.get_rng_state().clone()
    observed_states = []

    def replay(_model, _tokenizer, prompt_ids, row, **_kwargs):
        observed_states.append(torch.equal(torch.get_rng_state(), expected_state))
        return NativeRollout([int(prompt_ids[0])], [True], [-0.1], [0.5], "eos", ())

    monkeypatch.setattr(hf_generation, "generate_native_rollout", replay)
    rollouts = generate_native_rollout_batch(
        model,
        tokenizer,
        [([4], {"question": "a", "answer": "1"}), ([5], {"question": "b", "answer": "2"})],
        context_length=4,
        max_completion_tokens=3,
        temperature=1.0,
        rollout=RolloutConfig(executor="exact"),
    )
    assert len(rollouts) == 2
    assert observed_states == [True, True]


def test_native_rollout_fails_closed_when_call_boundary_is_ambiguous_or_disallowed():
    ambiguous = MultiTokenTokenizer("<think>\ncall:add(1,2) -> 3\n")
    compound = {"question": "Calculate (1+2)+0.", "answer": "3"}
    result = generate_native_rollout(
        CachedGenerationModel(),
        ambiguous,
        [1],
        compound,
        context_length=128,
        max_completion_tokens=32,
        temperature=1.0,
        rollout=RolloutConfig(executor="exact"),
        greedy=True,
    )
    assert result.termination_reason == "malformed_request"
    assert result.executor_tokens == 0

    disallowed = MultiTokenTokenizer("<think>\ncall:add(1,2) -> ")
    atomic = {"question": "Calculate 1 + 2.", "answer": "3"}
    denied = generate_native_rollout(
        CachedGenerationModel(),
        disallowed,
        [1],
        atomic,
        context_length=128,
        max_completion_tokens=32,
        temperature=1.0,
        rollout=RolloutConfig(executor="exact"),
        greedy=True,
    )
    assert denied.termination_reason == "disallowed_call"


class TinyCausalTable(torch.nn.Module):
    def __init__(self, vocab_size=9):
        super().__init__()
        self.table = torch.nn.Parameter(torch.randn(vocab_size, vocab_size))

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        return type("Output", (), {"logits": self.table[input_ids]})()


def test_chunked_grpo_gradients_match_full_batch_with_executor_observation_masks():
    prompts = ([1, 2], [2, 3, 4])
    group_rollouts = [
        (
            NativeRollout(
                [4, 5, 6], [True, False, True], [-0.4, 0.0, -0.6], [0.8, 0.0, 0.7], "eos", ()
            ),
            NativeRollout([5, 6], [True, True], [-0.3, -0.5], [0.9, 0.8], "eos", ()),
        ),
        (
            NativeRollout(
                [3, 4, 5, 6],
                [True, True, False, True],
                [-0.2, -0.3, 0.0, -0.4],
                [0.8, 0.9, 0.0, 0.6],
                "eos",
                (),
            ),
            NativeRollout(
                [4, 5, 6], [True, True, True], [-0.5, -0.4, -0.2], [0.7, 0.8, 0.9], "eos", ()
            ),
        ),
    ]
    groups = [PromptRollouts(prompt, rollouts) for prompt, rollouts in zip(prompts, group_rollouts)]
    rewards = torch.tensor([[1.0, 0.0], [0.1, 1.0]])
    advantages = group_relative_advantages(rewards)
    model = TinyCausalTable()
    current_rows, old_rows, entropy_rows, masks = [], [], [], []
    for prompt, rollouts in zip(prompts, group_rollouts):
        for rollout in rollouts:
            current, old, entropy, mask = native_policy_statistics(
                model, prompt, rollout, temperature=1.0
            )
            current_rows.append(current)
            old_rows.append(old)
            entropy_rows.append(entropy)
            masks.append(mask)
    max_tokens = max(len(row) for row in current_rows)
    pad = lambda values, fill: torch.stack(
        [torch.nn.functional.pad(row, (0, max_tokens - len(row)), value=fill) for row in values]
    )
    baseline = grpo_objective(
        pad(current_rows, 0).unsqueeze(1),
        pad(old_rows, 0).unsqueeze(1),
        advantages.reshape(-1, 1),
        pad(masks, False).unsqueeze(1),
        entropy=pad(entropy_rows, 0).unsqueeze(1),
    )
    baseline.loss.backward()
    expected_grad = model.table.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    metrics = chunked_grpo_backward(
        model,
        groups,
        advantages,
        temperature=1.0,
        clip_epsilon=0.2,
        chunk_size=2,
    )
    assert torch.allclose(model.table.grad, expected_grad, atol=1e-6, rtol=1e-6)
    assert metrics["policy_tokens"] == sum(int(mask.sum()) for mask in masks)
    assert metrics["loss"] == pytest.approx(float(baseline.loss.detach()), abs=1e-6)


def test_native_policy_entropy_remains_finite_when_extra_logits_are_masked():
    model = TinyCausalTable(vocab_size=9)
    rollout = NativeRollout([2, 3], [True, True], [-0.4, -0.5], [0.8, 0.7], "eos", ())
    _current, _old, entropy, _mask = native_policy_statistics(
        model, [1], rollout, temperature=1.0, vocab_size=7
    )
    assert torch.isfinite(entropy).all()
