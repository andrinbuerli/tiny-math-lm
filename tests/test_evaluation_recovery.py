from __future__ import annotations

import csv
import json

import pytest
import torch

from tinymath.checkpoint import (
    compatibility_metadata,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.eval.evaluate import evaluate_checkpoint
from tinymath.model.gpt import GPTConfig, TinyMathGPT


def _reasoning_fixture(workspace_tmpdir, questions=("1 + 1", "2 + 2")):
    bundle = workspace_tmpdir / "bundle"
    bundle.mkdir()
    rows = [
        {"question": question, "answer": answer, "category": "interpolate", "module": "arithmetic__add_or_sub"}
        for question, answer in zip(questions, ("2", "4"))
    ]
    tokenizer = CharacterTokenizer.from_texts(
        [format_prompt(row["question"], protocol="reasoning_v1") for row in rows]
        + ["+(1,1)\n1,1,0:2,0\n+(2,2)\n2,2,0:4,0\nA: 2\nA: 4"],
        protocol="reasoning_v1",
    )
    tokenizer.save(bundle / "char_tokenizer.json")
    (bundle / "test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    annotation_sha = "a" * 64
    manifest = {
        "context_length": 64,
        "annotation": {"protocol": "reasoning_v1", "annotation_sha256": annotation_sha},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size, context_length=64, n_layers=1,
        n_heads=2, d_model=16, d_ff=32,
    )
    model = TinyMathGPT(model_config)
    checkpoint = workspace_tmpdir / "sft" / "seed18" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    bundle_hash = __import__("hashlib").sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    compatibility = compatibility_metadata(
        protocol="reasoning_v1", annotation_sha256=annotation_sha,
        tokenizer_sha256=tokenizer.sha256, bundle_sha256=bundle_hash, context_length=64,
    )
    save_checkpoint(
        checkpoint,
        {
            "model": model.state_dict(), "model_config": model_config.__dict__,
            "tokenizer_sha256": tokenizer.sha256, "compatibility": compatibility,
            "step": 3, "training_arm": "sft",
        },
    )
    return bundle, checkpoint, tokenizer


def test_checkpoint_compatibility_rejects_each_identity_mismatch():
    expected = compatibility_metadata(
        protocol="reasoning_v1", annotation_sha256="ann", tokenizer_sha256="tok",
        bundle_sha256="bundle", context_length=512,
    )
    for field, other in (
        ("protocol", "answer_only"), ("annotation_sha256", "other-ann"),
        ("tokenizer_sha256", "other-tok"), ("bundle_sha256", "other-bundle"),
        ("context_length", 256),
    ):
        actual = dict(expected)
        actual[field] = other
        with pytest.raises(ValueError, match=field):
            validate_checkpoint_compatibility({"compatibility": actual}, expected)


def test_reasoning_evaluation_resumes_and_keeps_structural_predictions(workspace_tmpdir, monkeypatch):
    bundle, checkpoint, tokenizer = _reasoning_fixture(workspace_tmpdir)
    output = workspace_tmpdir / "evaluation"
    calls = []

    def interrupt_after_first(model, prompt, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        completion = tokenizer.encode_completion("2", scratchpad="+(1,1)\n1,1,0:2,0")
        return torch.tensor([prompt[0].tolist() + completion], device=prompt.device)

    monkeypatch.setattr(TinyMathGPT, "generate", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluate_checkpoint(
            checkpoint_path=checkpoint, bundle_dir=bundle, output_dir=output,
            split_names=["test"], max_new_tokens=2,
        )
    progress = json.loads((output / "progress.json").read_text())
    assert len(progress["scored"]["test"]) == 1
    assert (output / "test_predictions.csv").is_file()

    calls.clear()

    def complete(model, prompt, **kwargs):
        calls.append(1)
        answer = "4" if len(calls) == 1 else "2"
        scratchpad = "+(1,1)\n1,1,0:2,0" if answer == "2" else "+(2,2)\n2,2,0:4,0"
        completion = tokenizer.encode_completion(answer, scratchpad=scratchpad)
        return torch.tensor([prompt[0].tolist() + completion], device=prompt.device)

    monkeypatch.setattr(TinyMathGPT, "generate", complete)
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint, bundle_dir=bundle, output_dir=output,
        split_names=["test"], max_new_tokens=2,
    )
    assert len(calls) == 1  # first scored row was skipped
    assert result["splits"]["test"]["scored_examples"] == 2
    assert result["splits"]["test"]["accuracy"] == 1.0
    with (output / "test_predictions.csv").open(encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 2
    assert predictions[0]["scratchpad"] == "+(1,1)\n1,1,0:2,0"
    assert json.loads((output / "results.json").read_text())["protocol"] == "reasoning_v1"


def test_reasoning_progress_refuses_changed_split_identity(workspace_tmpdir, monkeypatch):
    bundle, checkpoint, tokenizer = _reasoning_fixture(workspace_tmpdir, questions=("1 + 1",))
    output = workspace_tmpdir / "evaluation"
    monkeypatch.setattr(
        TinyMathGPT, "generate",
        lambda model, prompt, **kwargs: torch.tensor(
            [prompt[0].tolist() + tokenizer.encode_completion("2", scratchpad="+(1,1)\n1,1,0:2,0")],
            device=prompt.device,
        ),
    )
    evaluate_checkpoint(
        checkpoint_path=checkpoint, bundle_dir=bundle, output_dir=output,
        split_names=["test"], max_new_tokens=2,
    )
    (bundle / "test.jsonl").write_text(
        json.dumps({"question": "3 + 3", "answer": "6", "category": "interpolate", "module": "arithmetic__add_or_sub"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different checkpoint, split, or protocol"):
        evaluate_checkpoint(
            checkpoint_path=checkpoint, bundle_dir=bundle, output_dir=output,
            split_names=["test"], max_new_tokens=2,
        )


def test_reasoning_generation_uses_context_allowance_and_invalidates_capped_output(workspace_tmpdir, monkeypatch):
    bundle, checkpoint, tokenizer = _reasoning_fixture(workspace_tmpdir, questions=("1 + 1",))
    monkeypatch.setattr(TinyMathGPT, "generate", lambda model, prompt, **kwargs: (
        torch.tensor([prompt[0].tolist() + [tokenizer.token_to_id["\n"]] * kwargs["max_new_tokens"]], device=prompt.device)
    ))
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint, bundle_dir=bundle, output_dir=workspace_tmpdir / "capped",
        split_names=["test"], max_new_tokens=1,
    )
    assert result["decoding"]["per_prompt_cap"] == "context_length - prompt_tokens"
    assert result["splits"]["test"]["capped_completions"] == 1
    assert result["splits"]["test"]["accuracy"] == 0.0
