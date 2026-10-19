import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.inference import (
    discover_inference_checkpoints,
    format_raw_text,
    format_tool_call,
    load_inference_bundle,
    sample_split_example,
    sample_split_question,
    stream_inference_events,
)
from tinymath.model.gpt import GPTConfig, TinyMathGPT


def _write_tokenizer(tokenizer: CharacterTokenizer, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 4,
                "protocol": tokenizer.protocol,
                "special_tokens": tokenizer.special_tokens,
                "id_to_token": tokenizer.id_to_token,
                "token_to_id": tokenizer.token_to_id,
                "sha256": tokenizer.sha256,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_discover_inference_checkpoints(tmp_path):
    run = tmp_path / "runs" / "exp"
    expected = run / "1m" / "sft" / "seed18" / "latest_inference.pt"
    expected.parent.mkdir(parents=True)
    expected.touch()
    (expected.parent / "latest.pt").touch()

    found = discover_inference_checkpoints(run)

    assert [(item.path, item.label) for item in found] == [(expected, "1m/sft/seed18")]
    assert discover_inference_checkpoints(tmp_path / "missing") == []


def test_sample_questions_from_train_validation_and_test(tmp_path):
    for filename, questions in {
        "sft_train.jsonl": ["training example A", "training example B"],
        "sft_validation.jsonl": ["validation example"],
        "test.jsonl": ["held-out example"],
    }.items():
        (tmp_path / filename).write_text(
            "".join(json.dumps({"question": question, "answer": "hidden"}) + "\n"
                    for question in questions),
            encoding="utf-8",
        )

    assert sample_split_question(
        "train", data_bundle=tmp_path, rng=random.Random(7)
    ) in {"training example A", "training example B"}
    assert sample_split_question("validation", data_bundle=tmp_path) == "validation example"
    assert sample_split_question("test", data_bundle=tmp_path) == "held-out example"
    with pytest.raises(ValueError, match="Unsupported split"):
        sample_split_question("rlvr", data_bundle=tmp_path)


def test_sample_split_example_keeps_reference_answer_and_scratchpad(tmp_path):
    example = {
        "question": "2 + 2",
        "answer": "4",
        "scratchpad": "step:add\n=>4",
    }
    (tmp_path / "test.jsonl").write_text(json.dumps(example) + "\n", encoding="utf-8")

    assert sample_split_example("test", data_bundle=tmp_path) == example


class ScriptedInferenceModel(torch.nn.Module):
    """Deterministic tiny model that scripts independent main and sidecar streams."""

    def __init__(self, tokenizer, main_completion, sidecar_completion):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(context_length=256)
        self.tokenizer = tokenizer
        self.scripts = {
            "Q: Calculate 1+2.\n": tokenizer.encode(main_completion, add_eos=True),
            "Q: Compute 1 + 2.\n": tokenizer.encode(sidecar_completion, add_eos=True),
        }

    def forward_with_cache(self, input_ids, past_key_values):
        if past_key_values is None:
            transcript = input_ids[0].tolist()
            prompt = self.tokenizer.decode(transcript)
            prompt_key = next(key for key in self.scripts if prompt.startswith(key))
            prefix = self.tokenizer.encode(prompt_key, add_bos=True)
            mode = prompt_key
        else:
            transcript, mode = past_key_values
            transcript = transcript + input_ids[0].tolist()
            prefix = self.tokenizer.encode(mode, add_bos=True)
        next_index = len(transcript) - len(prefix)
        target = self.scripts[mode]
        token_id = target[next_index] if next_index < len(target) else self.tokenizer.eos_id
        logits = torch.full(
            (1, input_ids.shape[1], self.tokenizer.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits[0, -1, token_id] = 100.0
        return logits, (transcript, mode)


def test_load_checkpoint_bundle_and_generate(tmp_path):
    root = tmp_path
    bundle = root / "data" / "processed" / "bundle"
    bundle.mkdir(parents=True)
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: Calculate 23 + 48.\n", "<scp>\n</scp>\nA: 71"] , protocol="reasoning_v3"
    )
    _write_tokenizer(tokenizer, bundle / "char_tokenizer.json")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        n_layers=1,
        n_heads=2,
        d_model=8,
        d_ff=16,
    )
    model = TinyMathGPT(config)
    checkpoint = root / "runs" / "exp" / "1m" / "sft" / "seed18" / "latest_inference.pt"
    checkpoint.parent.mkdir(parents=True)
    (checkpoint.parent / "resolved_config.json").write_text(
        json.dumps({"data_bundle": "data/processed/bundle"}), encoding="utf-8"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": config.__dict__,
            "tokenizer_sha256": tokenizer.sha256,
            "training_arm": "sft",
            "update": 12,
        },
        checkpoint,
    )

    loaded, loaded_tokenizer, metadata = load_inference_bundle(
        checkpoint, repository_root=root, device="cpu"
    )
    events = list(stream_inference_events(
        loaded,
        loaded_tokenizer,
        "Calculate 23 + 48.",
        max_new_tokens=2,
    ))

    assert metadata["training_arm"] == "sft"
    assert metadata["update"] == 12
    assert loaded.config.vocab_size == tokenizer.vocab_size
    assert sum(event["kind"] == "primary_token" for event in events) == 2


def test_primary_uses_the_parallel_lm_tool_response_without_exact_calculation():
    main = "<scp>\nstep:add\ncall:add(1,2) -> 9\n</scp>\nA: 9"
    sidecar = "<scp>\nstep:+(1,2)=9\n</scp>\nA: 9"
    tokenizer = CharacterTokenizer.from_texts(
        [
            "Q: Calculate 1+2.\n",
            "Q: Compute 1 + 2.\n",
            main,
            sidecar,
            "Compute 1 + 2.",
        ],
        protocol="reasoning_v3",
    )
    model = ScriptedInferenceModel(tokenizer, main, sidecar)

    events = list(stream_inference_events(
        model,
        tokenizer,
        "Calculate 1+2.",
        max_new_tokens=128,
    ))

    primary = "".join(event["text"] for event in events if event["kind"] == "primary_token")
    sidecar_output = "".join(event["text"] for event in events if event["kind"] == "sidecar_token")
    assert primary == main
    assert sidecar_output == sidecar
    assert any(event["kind"] == "tool_started" for event in events)
    call = next(event for event in events if event["kind"] == "tool_started")
    assert call["operation"] == "add"
    assert call["left"] == "1"
    assert call["right"] == "2"
    assert call["tool_tokens"] == len(tokenizer.encode("call:add(1,2) -> "))
    sidecar_start = next(event for event in events if event["kind"] == "sidecar_start")
    assert sidecar_start["call_id"] == call["call_id"]
    assert sidecar_start["input_tokens"] > 0
    primary_start = next(event for event in events if event["kind"] == "generation_start")
    assert primary_start["input_tokens"] == len(tokenizer.encode(
        "Q: Calculate 1+2.\n", add_bos=True
    ))
    assert any(
        event["kind"] == "tool_result"
        and event["result"] == "9"
        and event["source"] == "parallel_lm"
        for event in events
    )
    assert not any(event["kind"] == "tool_error" for event in events)


def test_sidecar_uses_the_same_configured_completion_limit():
    main = "<scp>\ncall:add(1,2) -> "
    sidecar = "x" * 100
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: Calculate 1+2.\n", "Q: Compute 1 + 2.\n", main, sidecar],
        protocol="reasoning_v3",
    )
    model = ScriptedInferenceModel(tokenizer, main, sidecar)

    events = list(
        stream_inference_events(model, tokenizer, "Calculate 1+2.", max_new_tokens=64)
    )

    streamed_sidecar = "".join(
        event["text"] for event in events if event["kind"] == "sidecar_token"
    )
    assert streamed_sidecar == sidecar[:64]


def test_formats_raw_completion_and_inline_tool_expression():
    raw = format_raw_text("<scp>\nstep:add numbers\nA: 1\n</scp>")

    assert "&lt;scp&gt;" in raw
    assert "step:add numbers" in raw
    assert "A: 1" in raw
    assert format_tool_call("add", "23", "48") == "add(23, 48)"


def test_stream_rejects_unknown_characters_and_nonempty_constraint():
    tokenizer = CharacterTokenizer.from_texts(["Q: Calculate 1+2.\n"], protocol="reasoning_v3")
    model = TinyMathGPT(GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=64,
        n_layers=1,
        n_heads=2,
        d_model=8,
        d_ff=16,
    ))
    with pytest.raises(ValueError, match="outside this checkpoint tokenizer"):
        list(stream_inference_events(model, tokenizer, "Calculate π."))
    with pytest.raises(ValueError, match="nonempty"):
        list(stream_inference_events(model, tokenizer, "   "))


def test_malformed_call_is_shown_but_never_starts_parallel_lm():
    main = "<scp>\nstep:try a function\ncall:power(1,2) -> \n</scp>\nA: 1"
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: Calculate 1+2.\n", "Q: Compute 1 + 2.\n", main, "1"],
        protocol="reasoning_v3",
    )
    model = ScriptedInferenceModel(tokenizer, main, "1")

    events = list(stream_inference_events(
        model, tokenizer, "Calculate 1+2.", max_new_tokens=128
    ))

    assert any(event["kind"] == "tool_error" for event in events)
    assert not any(event["kind"] == "tool_started" for event in events)
    assert not any(event["kind"] == "sidecar_token" for event in events)
