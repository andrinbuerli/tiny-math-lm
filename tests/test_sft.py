from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_completion, format_prompt
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.train.runner import train_sft
from tinymath.train.sft import build_completion_batch, completion_only_loss


def test_loss_mask_begins_at_first_answer_target_and_includes_eos():
    examples = [{"prompt": "Q: 1+1.\nA: ", "completion": "2"}]
    tokenizer = CharacterTokenizer.from_texts([examples[0]["prompt"] + "2"])
    _inputs, targets, mask = build_completion_batch(examples, tokenizer, context_length=32)
    prompt_length = len(tokenizer.encode(examples[0]["prompt"], add_bos=True))
    assert not mask[0, : prompt_length - 1].any()
    assert mask[0, prompt_length - 1 :].tolist() == [True, True]
    assert targets[0, -2].item() == tokenizer.token_to_id["2"]
    assert targets[0, -1].item() == tokenizer.eos_id


def test_reasoning_loss_mask_includes_boundary_tokens_and_answer_eos():
    prompt = "Q: Calculate 47 + 28.\n"
    example = {
        "prompt": prompt,
        "completion": "75",
        "scratchpad": "+(47,28)\n7,8,0:5,1\n4,2,1:7,0",
    }
    tokenizer = CharacterTokenizer.from_texts(
        [prompt, example["scratchpad"], "\nA: 75"], protocol="reasoning_v1"
    )
    _inputs, targets, mask = build_completion_batch([example], tokenizer, context_length=128)
    prompt_size = len(tokenizer.encode(prompt, add_bos=True))
    assert mask[0, : prompt_size - 1].sum().item() == 0
    assert mask[0, prompt_size - 1 :].all()
    assert targets[0, prompt_size - 1].item() == tokenizer.scp_id
    assert targets[0, -1].item() == tokenizer.eos_id


def test_completion_loss_ignores_prompt_and_padding_logits():
    examples = [
        {"prompt": "Q: 1\nA: ", "completion": "2"},
        {"prompt": "Q: 2\nA: ", "completion": "30"},
    ]
    tokenizer = CharacterTokenizer.from_texts([e["prompt"] + e["completion"] for e in examples])
    _inputs, targets, mask = build_completion_batch(examples, tokenizer, context_length=32)
    logits = torch.zeros((*targets.shape, tokenizer.vocab_size))
    loss = completion_only_loss(logits, targets, mask)
    logits[~mask] = 10000
    assert torch.isfinite(loss)
    assert torch.equal(loss, completion_only_loss(logits, targets, mask))


def test_small_model_overfits_one_completion_example():
    torch.manual_seed(23)
    example = {"prompt": "Q: 1 + 1\nA: ", "completion": "2"}
    tokenizer = CharacterTokenizer.from_texts([example["prompt"] + example["completion"]])
    inputs, targets, mask = build_completion_batch([example], tokenizer, context_length=64)
    model = TinyMathGPT(
        GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=64,
            n_layers=1,
            n_heads=2,
            d_model=16,
            d_ff=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)
    initial_loss = completion_only_loss(model(inputs), targets, mask).item()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = completion_only_loss(model(inputs), targets, mask)
        loss.backward()
        optimizer.step()
    final_loss = completion_only_loss(model(inputs), targets, mask).item()
    assert final_loss < initial_loss * 0.25


def test_long_example_is_rejected_without_truncation():
    tokenizer = CharacterTokenizer.from_texts(["Q: long\nA: answer"])
    try:
        build_completion_batch(
            [{"prompt": "Q: long\nA: ", "completion": "answer"}], tokenizer, context_length=4
        )
    except ValueError as error:
        assert "never silently truncated" in str(error)
    else:
        raise AssertionError("overlength examples must be excluded or rejected explicitly")


def test_sft_loop_trains_and_saves_reloadable_checkpoint(workspace_tmpdir):
    bundle = workspace_tmpdir / "bundle"
    run_dir = workspace_tmpdir / "run"
    bundle.mkdir()
    rows = [
        {"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"},
        {"question": "2 + 1", "answer": "3", "module": "arithmetic__add_or_sub"},
    ]
    tokenizer = CharacterTokenizer.from_texts(
        format_prompt(row["question"]) + format_completion(row["answer"]) for row in rows
    )
    tokenizer.save(bundle / "char_tokenizer.json")
    for filename, examples in (
        ("sft_train.jsonl", rows),
        ("sft_validation.jsonl", rows[:1]),
        ("test.jsonl", rows[:1]),
    ):
        (bundle / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
        )
    (bundle / "manifest.json").write_text(
        json.dumps({"context_length": 64, "split_manifest_sha256": "fixture"}),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        seed=17,
        model={
            "vocab_size": None,
            "context_length": 64,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 16,
            "d_ff": 32,
            "dropout": 0.0,
        },
        trainer=SimpleNamespace(
            name="sft",
            bf16=False,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=2,
            batch_size=2,
            gradient_accumulation=1,
            eval_every=1,
            save_every=1,
            warmup_steps=0,
            grad_clip=1.0,
        ),
        tracking=SimpleNamespace(log_every=1),
        qualitative=SimpleNamespace(every_steps=1, examples_per_module=2),
    )

    class Run:
        def __init__(self):
            self.logged = []

        def log(self, values, **_kwargs):
            self.logged.append(values)

    run = Run()
    result = train_sft(config=config, bundle_dir=bundle, run_dir=run_dir, run=run)
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert result["step"] == 2
    assert checkpoint["step"] == 2
    assert checkpoint["tokenizer_sha256"] == tokenizer.sha256
    assert checkpoint["compatibility"]["protocol"] == "answer_only"
    assert checkpoint["compatibility"]["context_length"] == 64
    assert checkpoint["examples_drawn"] == 4
    assert checkpoint["nonpadding_input_tokens"] > 0
    assert checkpoint["padded_compute_positions"] == checkpoint["tokens_seen"]
    assert checkpoint["supervised_tokens"] > 0
    assert checkpoint["data_generator_state"].numel() > 0
    assert checkpoint["model"]
    assert len(run.logged) == 2
    assert "qualitative/examples" in run.logged[-1]
    assert run.logged[-1]["train/examples_drawn"] == 4
    assert run.logged[-1]["train/supervised_tokens"] > 0
    assert "train/greedy_accuracy" in run.logged[-1]
    assert "val/greedy_accuracy" in run.logged[-1]
    assert "test/greedy_accuracy" in run.logged[-1]
    assert result["qualitative_examples"] == 4
    inference = torch.load(result["inference_checkpoint"], map_location="cpu", weights_only=False)
    assert inference["step"] == 2
    assert inference["training_arm"] == "sft"
    assert inference["tokenizer_sha256"] == tokenizer.sha256
    assert inference["compatibility"] == checkpoint["compatibility"]
    assert inference["model"]
    assert Path(result["metrics_path"]).exists()
    with Path(result["metrics_path"]).open(encoding="utf-8") as existing:
        fieldnames = next(csv.reader(existing))
    assert {
        "examples_drawn", "nonpadding_input_tokens", "padded_compute_positions", "supervised_tokens"
    }.issubset(fieldnames)
    with Path(result["metrics_path"]).open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {"optimizer_step": 2, "train/loss": "duplicate"}
        )
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {"optimizer_step": 4, "train/loss": "stale"}
        )
    config.trainer.max_steps = 3
    config.rollout = {
        "executor": "exact",
        "max_calls": 32,
        "max_scalar_chars": 128,
        "format_weight": 0.1,
        "success_weight": 0.9,
        "scorer_version": 2,
        "executor_version": 1,
        "rollout_version": 2,
        "token_accounting": "transcript_includes_executor_v1",
        "delegation_policy": "proper_subexpression_v1",
    }
    resumed = train_sft(config=config, bundle_dir=bundle, run_dir=run_dir, run=run)
    resumed_checkpoint = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert resumed["step"] == resumed_checkpoint["step"] == 3
    resume_qualitative = run.logged[-2]
    assert resume_qualitative["optimizer_step"] == 2
    assert resume_qualitative["qualitative/resume_checkpoint"] is True
    assert resume_qualitative["qualitative/legacy_rollout_fallback"] is True
    assert "qualitative/examples" in resume_qualitative
    assert run.logged[-1]["optimizer_step"] == 3
    with Path(resumed["metrics_path"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert [int(row["optimizer_step"]) for row in rows] == [1, 2, 3]
        assert rows[1]["train/loss"] != "duplicate"


def test_continued_sft_resumes_its_own_checkpoint_and_records_source(workspace_tmpdir):
    bundle = workspace_tmpdir / "bundle"
    source_dir = workspace_tmpdir / "sft"
    control_dir = workspace_tmpdir / "continued_sft"
    bundle.mkdir()
    rows = [
        {"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"},
        {"question": "2 + 1", "answer": "3", "module": "arithmetic__add_or_sub"},
    ]
    tokenizer = CharacterTokenizer.from_texts(
        format_prompt(row["question"]) + format_completion(row["answer"]) for row in rows
    )
    tokenizer.save(bundle / "char_tokenizer.json")
    for filename, examples in (
        ("sft_train.jsonl", rows),
        ("sft_validation.jsonl", rows[:1]),
        ("test.jsonl", rows[:1]),
    ):
        (bundle / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
        )
    config = SimpleNamespace(
        seed=19,
        model={
            "vocab_size": None,
            "context_length": 64,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 16,
            "d_ff": 32,
            "dropout": 0.0,
        },
        trainer=SimpleNamespace(
            name="sft",
            bf16=False,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=2,
            batch_size=2,
            gradient_accumulation=1,
            eval_every=1,
            save_every=1,
            warmup_steps=0,
            grad_clip=1.0,
        ),
        tracking=SimpleNamespace(log_every=10),
        qualitative=SimpleNamespace(every_steps=100, examples_per_module=1),
    )

    class Run:
        def __init__(self):
            self.summary = {}

        def log(self, _values, **_kwargs):
            pass

    run = Run()
    source_result = train_sft(config=config, bundle_dir=bundle, run_dir=source_dir, run=run)
    source_checkpoint = Path(source_result["checkpoint"]).resolve()
    config.trainer.name = "continued_sft"
    first = train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=control_dir,
        run=run,
        init_checkpoint=source_checkpoint,
    )
    first_payload = torch.load(first["checkpoint"], map_location="cpu", weights_only=False)
    assert first["step"] == first_payload["step"] == 2
    assert first_payload["training_arm"] == "continued_sft"
    assert first_payload["source_sft_checkpoint"] == str(source_checkpoint)
    assert first_payload["source_sft_tokens_seen"] == source_result["tokens_seen"]
    assert first["additional_estimated_flops"] == 6 * first["parameters"] * first["tokens_seen"]

    config.trainer.max_steps = 3
    resumed = train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=control_dir,
        run=run,
        init_checkpoint=source_checkpoint,
    )
    resumed_payload = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert resumed["step"] == resumed_payload["step"] == 3
    assert resumed_payload["source_sft_checkpoint"] == str(source_checkpoint)
    assert resumed_payload["source_sft_tokens_seen"] == source_result["tokens_seen"]
    assert resumed["additional_estimated_flops"] == 6 * resumed["parameters"] * resumed["tokens_seen"]
    with Path(resumed["metrics_path"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert [int(row["optimizer_step"]) for row in rows] == [1, 2, 3]
        assert int(float(rows[-1]["additional_estimated_flops"])) == resumed[
            "additional_estimated_flops"
        ]

    budget_dir = workspace_tmpdir / "continued_sft_budget"
    with Path(first["metrics_path"]).open(encoding="utf-8") as handle:
        first_metrics = list(csv.DictReader(handle))
    first_update_positions = int(first_metrics[0]["tokens_seen"])
    budget_target = 6 * first["parameters"] * first_update_positions
    config.trainer.compute_budget_flops = budget_target
    config.trainer.max_steps = 10
    config.trainer.save_every = 1000
    budgeted = train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=budget_dir,
        run=run,
        init_checkpoint=source_checkpoint,
    )
    latest = torch.load(budgeted["checkpoint"], map_location="cpu", weights_only=False)
    inference = torch.load(budgeted["inference_checkpoint"], map_location="cpu", weights_only=False)
    assert budgeted["step"] == latest["step"] == inference["step"] == 1
    assert budgeted["compute_budget_reached"] is True
    assert budgeted["compute_budget_fraction"] == 1.0
    assert latest["compute_budget_flops"] == inference["compute_budget_flops"] == budget_target
    assert latest["compute_accounting"] == inference["compute_accounting"] == "dense_padded_positions_v1"
    assert latest["compute_spent_flops"] == budget_target
    assert budgeted["padded_compute_positions"] == first_update_positions
    with pytest.raises(ValueError, match="compute budget does not match"):
        config.trainer.compute_budget_flops = 2
        train_sft(
            config=config,
            bundle_dir=bundle,
            run_dir=budget_dir,
            run=run,
            init_checkpoint=source_checkpoint,
        )

    # A budgeted continuation resumed after step 1 must exactly match an
    # uninterrupted three-step run, including compute-based LR decay.
    config.trainer.compute_budget_flops = (
        6 * first["parameters"] * (int(first_metrics[1]["tokens_seen"]) + 10 * first_update_positions)
    )
    config.trainer.warmup_steps = 1
    config.trainer.save_every = 1000
    resumable_dir = workspace_tmpdir / "continued_sft_resumable_budget"
    config.trainer.max_steps = 1
    torch.manual_seed(12345)
    train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=resumable_dir,
        run=run,
        init_checkpoint=source_checkpoint,
    )
    config.trainer.max_steps = 3
    resumed_budget = train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=resumable_dir,
        run=run,
        init_checkpoint=source_checkpoint,
    )
    uninterrupted_dir = workspace_tmpdir / "continued_sft_uninterrupted_budget"
    torch.manual_seed(12345)
    uninterrupted = train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=uninterrupted_dir,
        run=run,
        init_checkpoint=source_checkpoint,
    )
    resumed_payload = torch.load(resumed_budget["checkpoint"], map_location="cpu", weights_only=False)
    uninterrupted_payload = torch.load(
        uninterrupted["checkpoint"], map_location="cpu", weights_only=False
    )
    assert resumed_budget["step"] == uninterrupted["step"] == 3
    assert resumed_budget["compute_budget_reached"] is False
    assert all(
        torch.equal(resumed_payload["model"][key], uninterrupted_payload["model"][key])
        for key in resumed_payload["model"]
    )
    with Path(uninterrupted["metrics_path"]).open(encoding="utf-8") as handle:
        schedule_rows = list(csv.DictReader(handle))
    assert float(schedule_rows[2]["learning_rate"]) < float(schedule_rows[1]["learning_rate"])
