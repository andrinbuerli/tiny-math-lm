from __future__ import annotations

import json

from tinymath.checkpoint import save_checkpoint
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.eval.evaluate import evaluate_checkpoint
from tinymath.model.gpt import GPTConfig, TinyMathGPT


def test_greedy_evaluation_writes_linked_predictions_and_metrics(workspace_tmpdir):
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
    checkpoint = workspace_tmpdir / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        {
            "model": model.state_dict(),
            "model_config": model_config.__dict__,
            "tokenizer_sha256": tokenizer.sha256,
            "step": 4,
        },
    )
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        bundle_dir=bundle,
        output_dir=output,
        split_names=["test"],
        max_new_tokens=3,
    )
    assert result["checkpoint_step"] == 4
    assert result["splits"]["test"]["examples"] == 1
    assert (output / "test_predictions.csv").is_file()
    assert json.loads((output / "results.json").read_text())["tokenizer_sha256"] == tokenizer.sha256
