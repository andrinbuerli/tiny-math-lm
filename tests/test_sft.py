from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert checkpoint["data_generator_state"].numel() > 0
    assert checkpoint["model"]
    assert len(run.logged) == 2
    assert "qualitative/examples" in run.logged[-1]
    assert "train/greedy_accuracy" in run.logged[-1]
    assert "val/greedy_accuracy" in run.logged[-1]
    assert "test/greedy_accuracy" in run.logged[-1]
    assert result["qualitative_examples"] == 4
    inference = torch.load(result["inference_checkpoint"], map_location="cpu", weights_only=False)
    assert inference["step"] == 2
    assert inference["training_arm"] == "sft"
    assert inference["tokenizer_sha256"] == tokenizer.sha256
    assert inference["model"]
    assert Path(result["metrics_path"]).exists()
    with Path(result["metrics_path"]).open(encoding="utf-8") as existing:
        fieldnames = next(csv.reader(existing))
    with Path(result["metrics_path"]).open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {"optimizer_step": 2, "train/loss": "duplicate"}
        )
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {"optimizer_step": 4, "train/loss": "stale"}
        )
    config.trainer.max_steps = 3
    resumed = train_sft(config=config, bundle_dir=bundle, run_dir=run_dir, run=run)
    resumed_checkpoint = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert resumed["step"] == resumed_checkpoint["step"] == 3
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
    with Path(resumed["metrics_path"]).open(encoding="utf-8") as handle:
        assert [int(row["optimizer_step"]) for row in csv.DictReader(handle)] == [1, 2, 3]
