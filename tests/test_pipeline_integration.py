from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tiny_math_lm.data.char_tokenizer import CharacterTokenizer
from tiny_math_lm.data.prompts import format_completion, format_prompt
from tiny_math_lm.eval.evaluate import evaluate_checkpoint
from tiny_math_lm.train import grpo_runner, runner
from tiny_math_lm.train.grpo import Rollouts


def test_sft_grpo_checkpoint_evaluation_pipeline(workspace_tmpdir, monkeypatch):
    torch.manual_seed(29)
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    examples = [
        {"question": "1 + 1", "answer": "2", "module": "arithmetic__add_or_sub"},
        {"question": "2 + 1", "answer": "3", "module": "arithmetic__add_or_sub"},
    ]
    tokenizer = CharacterTokenizer.from_texts(
        format_prompt(row["question"]) + format_completion(row["answer"])
        for row in examples
    )
    tokenizer.save(bundle / "char_tokenizer.json")
    for filename, rows in (
        ("sft_train.jsonl", examples),
        ("sft_validation.jsonl", examples[:1]),
        ("test.jsonl", examples[:1]),
        ("rlvr.jsonl", examples[:1]),
    ):
        (bundle / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def fake_qualitative_table(**_kwargs):
        return "unit qualitative table", 3, {"train": 0.2, "validation": 0.3, "test": 0.4}

    monkeypatch.setattr(runner, "build_qualitative_table", fake_qualitative_table)
    monkeypatch.setattr(grpo_runner, "build_qualitative_table", fake_qualitative_table)

    class Run:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, values, **_kwargs):
            self.logged.append(values)

    run = Run()
    config = SimpleNamespace(
        seed=29,
        model={
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
            max_steps=1,
            batch_size=2,
            gradient_accumulation=1,
            eval_every=1,
            save_every=1,
            warmup_steps=0,
            grad_clip=1.0,
        ),
        tracking=SimpleNamespace(log_every=1),
        qualitative=SimpleNamespace(every_steps=5000, examples_per_module=1),
        eval=SimpleNamespace(max_new_tokens=4),
        rlvr=SimpleNamespace(
            updates=1,
            prompts_per_batch=1,
            group_size=2,
            temperature=1.0,
            max_completion_tokens=2,
            clip_epsilon=0.2,
            kl_coefficient=0.0,
            learning_rate=1e-2,
        ),
    )

    sft_result = runner.train_sft(
        config=config,
        bundle_dir=bundle,
        run_dir=workspace_tmpdir / "sft",
        run=run,
    )
    sft_checkpoint = Path(sft_result["checkpoint"])
    sft_payload = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
    assert sft_payload["step"] == 1
    assert sft_payload["training_arm"] == "sft"

    def fixed_rollouts(model, prompt_ids, *, temperature, **_kwargs):
        completion_ids = torch.tensor(
            [[
                [tokenizer.token_to_id["2"], tokenizer.eos_id],
                [tokenizer.token_to_id["3"], tokenizer.eos_id],
            ]],
            device=prompt_ids.device,
        )
        completion_mask = torch.ones_like(completion_ids, dtype=torch.bool)
        with torch.no_grad():
            old_log_probs = grpo_runner.completion_log_probs(
                model, prompt_ids, completion_ids, temperature=temperature
            )
        return Rollouts(
            completion_ids,
            completion_mask,
            old_log_probs,
            torch.ones_like(old_log_probs),
        )

    monkeypatch.setattr(grpo_runner, "sample_group_rollouts", fixed_rollouts)
    config.trainer.bf16 = False
    config.trainer.save_every = 1
    grpo_result = grpo_runner.train_grpo(
        config=config,
        bundle_dir=bundle,
        init_checkpoint=sft_checkpoint,
        run_dir=workspace_tmpdir / "grpo",
        run=run,
    )
    grpo_payload = torch.load(
        grpo_result["checkpoint"], map_location="cpu", weights_only=False
    )
    assert grpo_payload["update"] == 1
    assert grpo_payload["training_arm"] == "grpo"
    assert grpo_payload["source_sft_checkpoint"] == str(sft_checkpoint)
    inference_payload = torch.load(
        grpo_result["inference_checkpoint"], map_location="cpu", weights_only=False
    )
    assert inference_payload["source_sft_checkpoint"] == str(sft_checkpoint)

    evaluation = evaluate_checkpoint(
        checkpoint_path=Path(grpo_result["inference_checkpoint"]),
        bundle_dir=bundle,
        output_dir=workspace_tmpdir / "evaluation",
        split_names=["test"],
        max_new_tokens=4,
        max_examples_per_split=1,
    )
    assert evaluation["training_arm"] == "grpo"
    assert evaluation["source_sft_checkpoint"] == str(sft_checkpoint)
    assert evaluation["splits"]["test"]["examples"] == 1
    assert Path(workspace_tmpdir / "evaluation" / "results.json").is_file()
