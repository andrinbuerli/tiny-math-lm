from __future__ import annotations

import json

import pytest

from tiny_math_lm.checkpoint import save_checkpoint
from tiny_math_lm.data.char_tokenizer import CharacterTokenizer
from tiny_math_lm.data.prompts import format_prompt
from tiny_math_lm.eval.evaluate import _balanced_limit, evaluate_checkpoint
from tiny_math_lm.model.gpt import GPTConfig, TinyMathGPT


@pytest.mark.parametrize(
    ("greedy", "temperature", "strategy", "effective_temperature"),
    ((True, 0.8, "greedy", 0.0), (False, 0.6, "sampled", 0.6)),
)
def test_evaluation_writes_linked_predictions_and_decoding_metadata(
    workspace_tmpdir, monkeypatch, greedy, temperature, strategy, effective_temperature
):
    bundle = workspace_tmpdir / "bundle"
    output = workspace_tmpdir / "eval"
    bundle.mkdir()
    question = "1 + 1"
    tokenizer = CharacterTokenizer.from_texts([format_prompt(question) + "23"])
    tokenizer.save(bundle / "char_tokenizer.json")
    (bundle / "test.jsonl").write_text(
        json.dumps(
            {
                "question": question,
                "answer": "2",
                "category": "interpolate",
                "module": "arithmetic__add_or_sub",
            }
        ) + "\n",
        encoding="utf-8",
    )
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size, context_length=64, n_layers=1,
        n_heads=2, d_model=16, d_ff=32,
    )
    model = TinyMathGPT(model_config)
    checkpoint = workspace_tmpdir / "continued_sft" / "seed18" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    source_checkpoint = "/workspace/runs/source/sft/seed18/best.pt"
    (checkpoint.parent / "resolved_config.json").write_text(
        json.dumps({"init_checkpoint": source_checkpoint}), encoding="utf-8"
    )
    save_checkpoint(
        checkpoint,
        {
            "model": model.state_dict(),
            "model_config": model_config.__dict__,
            "tokenizer_sha256": tokenizer.sha256,
            "step": 4,
        },
    )
    generation_temperatures = []
    original_generate = TinyMathGPT.generate

    def record_temperature(model, *args, **kwargs):
        generation_temperatures.append(kwargs["temperature"])
        return original_generate(model, *args, **kwargs)

    monkeypatch.setattr(TinyMathGPT, "generate", record_temperature)
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        bundle_dir=bundle,
        output_dir=output,
        split_names=["test"],
        max_new_tokens=3,
        greedy=greedy,
        temperature=temperature,
    )
    assert result["checkpoint_step"] == 4
    assert result["training_arm"] == "continued_sft"
    assert result["source_sft_checkpoint"] == source_checkpoint
    assert result["decoding"] == {
        "strategy": strategy,
        "temperature": effective_temperature,
        "max_new_tokens": 3,
        "context_length": 64,
        "per_prompt_cap": "min(max_new_tokens, context_length - prompt_tokens)",
    }
    assert result["splits"]["test"]["examples"] == 1
    assert (output / "test_predictions.csv").is_file()
    saved_result = json.loads((output / "results.json").read_text())
    assert saved_result["tokenizer_sha256"] == tokenizer.sha256
    assert saved_result["decoding"] == result["decoding"]
    assert generation_temperatures == [effective_temperature]

    (bundle / "test.jsonl").write_text("", encoding="utf-8")
    empty_result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        bundle_dir=bundle,
        output_dir=workspace_tmpdir / "empty-eval",
        split_names=["test"],
        max_new_tokens=3,
    )["splits"]["test"]
    assert empty_result["examples"] == 0
    assert empty_result["accuracy"] is None
    assert empty_result["format_validity"] is None
    assert empty_result["mean_output_tokens"] is None


def test_sampled_evaluation_requires_positive_temperature():
    with pytest.raises(ValueError, match="positive temperature"):
        evaluate_checkpoint(
            checkpoint_path=None,
            bundle_dir=None,
            output_dir=None,
            split_names=[],
            max_new_tokens=1,
            greedy=False,
            temperature=0.0,
        )


def test_bounded_evaluation_subsets_round_robin_across_modules():
    examples = [
        {"module": module, "question": f"{module}-{index}"}
        for module in ("a", "b", "c")
        for index in range(5)
    ]
    selected = _balanced_limit(examples, 8)
    counts = {module: sum(row["module"] == module for row in selected) for module in ("a", "b", "c")}
    assert counts == {"a": 3, "b": 3, "c": 2}
