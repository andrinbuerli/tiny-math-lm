from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_completion, format_prompt
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
    for filename, examples in (("sft_train.jsonl", rows), ("sft_validation.jsonl", rows[:1])):
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
    assert Path(result["metrics_path"]).exists()
    config.trainer.max_steps = 3
    resumed = train_sft(config=config, bundle_dir=bundle, run_dir=run_dir, run=run)
    resumed_checkpoint = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert resumed["step"] == resumed_checkpoint["step"] == 3
