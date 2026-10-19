"""Checkpoint discovery, data sampling, and streamed TinyMath inference."""

from __future__ import annotations

import html
import json
import queue
import random
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.rewards.calls import ArithmeticCall, CallError, execute_call


@dataclass(frozen=True)
class InferenceCheckpoint:
    path: Path
    label: str


SPLIT_FILES = {
    "train": "sft_train.jsonl",
    "validation": "sft_validation.jsonl",
    "test": "test.jsonl",
}
_SCALAR = r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?"
_CALL_REQUEST = re.compile(rf"call:(?:add|subtract|multiply|divide)\({_SCALAR},{_SCALAR}\) -> ")


def discover_inference_checkpoints(run_root: str | Path) -> list[InferenceCheckpoint]:
    """Find lightweight inference snapshots under an experiment run directory."""
    root = Path(run_root).expanduser().resolve()
    if not root.is_dir():
        return []
    snapshots = []
    for path in root.rglob("latest_inference.pt"):
        relative = path.relative_to(root)
        snapshots.append(InferenceCheckpoint(path=path, label=str(relative.parent)))
    return sorted(snapshots, key=lambda item: item.label)


def _resolve_bundle_path(value: str, repository_root: Path) -> Path:
    repository_root = repository_root.resolve()
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.exists():
        resolved = candidate.resolve()
        if repository_root not in resolved.parents and resolved != repository_root:
            raise ValueError("Tokenizer/data bundle must be inside the repository workspace")
        return resolved
    # Saved configs use /workspace paths in the GPU container. Resolve their
    # workspace-relative suffix against this checkout for local UI use.
    if candidate.is_absolute() and candidate.parts[:2] == ("/", "workspace"):
        candidate = Path(*candidate.parts[2:])
    resolved = (repository_root / candidate).resolve()
    if repository_root not in resolved.parents and resolved != repository_root:
        raise ValueError("Tokenizer/data bundle must be inside the repository workspace")
    return resolved


def _run_metadata(checkpoint_path: Path, repository_root: Path) -> tuple[dict[str, Any], Path]:
    config_path = checkpoint_path.parent / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run configuration beside checkpoint: {config_path}")
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle_value = run_config.get("data_bundle")
    if not bundle_value:
        raise ValueError("Run configuration does not identify its tokenizer data_bundle")
    bundle = _resolve_bundle_path(str(bundle_value), repository_root)
    return run_config, bundle


def load_inference_bundle(
    checkpoint_path: str | Path,
    *,
    repository_root: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[TinyMathGPT, CharacterTokenizer, dict[str, Any]]:
    """Load model weights, tokenizer and run metadata without optimizer state."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    root = Path(repository_root).resolve()
    if root not in checkpoint_path.parents:
        raise ValueError("Checkpoint must be inside the repository workspace")
    run_config, bundle = _run_metadata(checkpoint_path, root)
    tokenizer = CharacterTokenizer.load(bundle / "char_tokenizer.json")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_values = state.get("model_config")
    weights = state.get("model")
    if not isinstance(model_values, dict) or not isinstance(weights, dict):
        raise TypeError("Checkpoint is not a TinyMath inference snapshot")
    if int(model_values.get("vocab_size", -1)) != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocabulary size does not match its tokenizer")
    expected_hash = state.get("tokenizer_sha256")
    if expected_hash and expected_hash != tokenizer.sha256:
        raise ValueError("Checkpoint tokenizer hash does not match the saved tokenizer")
    model = TinyMathGPT(GPTConfig.from_mapping(model_values))
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, {
        "protocol": tokenizer.protocol,
        "training_arm": state.get("training_arm", "unknown"),
        "update": state.get("update"),
        "model_config": model_values,
        "checkpoint": str(checkpoint_path),
        "data_bundle": str(bundle),
        "run_config": run_config,
    }


def sample_split_question(
    split: str,
    *,
    data_bundle: str | Path,
    rng: random.Random | None = None,
) -> str:
    """Select one question uniformly from a JSONL split using reservoir sampling."""
    try:
        filename = SPLIT_FILES[split]
    except KeyError as exc:
        raise ValueError(f"Unsupported split: {split}") from exc
    path = Path(data_bundle) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    chooser = rng or random.SystemRandom()
    selected = None
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                continue
            count += 1
            if chooser.randrange(count) == 0:
                selected = question
    if selected is None:
        raise ValueError(f"No question rows found in {path}")
    return selected


def unsupported_characters(tokenizer: CharacterTokenizer, question: str) -> list[str]:
    """Return unsupported Unicode code points in first-seen order."""
    return list(dict.fromkeys(character for character in question if character not in tokenizer.token_to_id))


def _escape_markdown(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>])", r"\\\1", escaped)


def format_raw_text(text: str) -> str:
    """Safely render raw model text while retaining visible protocol markers."""
    return "  \n".join(_escape_markdown(line) for line in text.splitlines())


def format_tool_call(operation: str, left: str, right: str) -> str:
    """Render a tool call as plain text."""
    return f"{operation}({left}, {right})"


def _sample_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)


def _plain_task(call: ArithmeticCall) -> str:
    symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[call.operation]
    return f"Compute {call.left} {symbol} {call.right}."


def _stream_plain_task(
    model: TinyMathGPT,
    tokenizer: CharacterTokenizer,
    call: ArithmeticCall,
    *,
    temperature: float,
    max_new_tokens: int,
    call_id: int,
    events: queue.Queue,
) -> None:
    """Generate an independent, tool-disabled sidecar answer into the event queue."""
    try:
        prompt_ids = tokenizer.encode(
            format_prompt(_plain_task(call), protocol=tokenizer.protocol), add_bos=True
        )
        events.put({
            "kind": "sidecar_start", "call_id": call_id, "task": _plain_task(call),
            "input_tokens": len(prompt_ids),
        })
        available = model.config.context_length - len(prompt_ids)
        if available <= 0:
            raise ValueError("Sidecar prompt exceeds the checkpoint context length")
        pending = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
        cache = None
        for _ in range(min(max_new_tokens, available)):
            logits, cache = model.forward_with_cache(pending, cache)
            token = _sample_logits(logits[:, -1], temperature)
            token_id = int(token.item())
            if token_id == tokenizer.eos_id:
                break
            piece = tokenizer.decode([token_id], skip_special=False)
            if piece:
                events.put({"kind": "sidecar_token", "call_id": call_id, "text": piece})
            pending = token
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        events.put({"kind": "sidecar_error", "call_id": call_id, "text": str(exc)})
    finally:
        events.put({"kind": "sidecar_done", "call_id": call_id})


def _primary_generation(
    model: TinyMathGPT,
    tokenizer: CharacterTokenizer,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    events: queue.Queue,
) -> None:
    """Incrementally generate the main trace and splice exact tool observations."""
    prompt_ids = tokenizer.encode(
        format_prompt(question.strip(), protocol=tokenizer.protocol), add_bos=True
    )
    if len(prompt_ids) >= model.config.context_length:
        raise ValueError("Question leaves no room for a completion in the model context")
    device = next(model.parameters()).device
    events.put({"kind": "generation_start", "input_tokens": len(prompt_ids)})
    pending = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = None
    generated_count = 0
    line = ""
    in_scratchpad = False
    seen = ""
    sidecar_futures = []
    call_index = 0

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="tinymath-tool") as executor:
        while generated_count < max_new_tokens:
            context_left = model.config.context_length - len(prompt_ids) - generated_count
            if context_left <= 0:
                break
            logits, cache = model.forward_with_cache(pending, cache)
            token = _sample_logits(logits[:, -1], temperature)
            token_id = int(token.item())
            generated_count += 1
            if token_id == tokenizer.eos_id:
                break
            piece = tokenizer.decode([token_id], skip_special=False)
            if not piece:
                pending = token
                continue
            events.put({
                "kind": "primary_token", "text": piece, "source": "model",
                "region": "reasoning" if in_scratchpad else "completion",
            })
            seen += piece
            if not in_scratchpad and seen.endswith("<scp>\n"):
                in_scratchpad = True
            elif in_scratchpad and seen.endswith("\n</scp>"):
                in_scratchpad = False
                line = ""
            if in_scratchpad:
                if piece == "\n":
                    line = ""
                else:
                    line += piece
                    if line.startswith("call:") and line.endswith(" -> "):
                        if _CALL_REQUEST.fullmatch(line) is None:
                            events.put({
                                "kind": "tool_error",
                                "text": "malformed_request",
                                "request": line,
                            })
                        else:
                            call_index += 1
                            call_id = call_index
                            operation, operands = line[5:].split("(", 1)
                            left, right = operands[:-5].split(",", 1)
                            sidecar_call = ArithmeticCall(operation, left, right, "")
                            events.put({
                                "kind": "tool_started",
                                "call_id": call_id,
                                "request": line,
                                "tool_tokens": len(tokenizer.encode(line)),
                                "operation": operation,
                                "left": left,
                                "right": right,
                            })
                            tool_future = executor.submit(execute_call, line)
                            sidecar = executor.submit(
                                _stream_plain_task,
                                model,
                                tokenizer,
                                sidecar_call,
                                temperature=temperature,
                                max_new_tokens=max_new_tokens,
                                call_id=call_id,
                                events=events,
                            )
                            sidecar_futures.append(sidecar)
                            try:
                                exact_call = tool_future.result()
                            except CallError as exc:
                                events.put({
                                    "kind": "tool_error",
                                    "call_id": call_id,
                                    "text": str(exc),
                                    "request": line,
                                })
                                return
                            result_ids = tokenizer.encode(exact_call.result)
                            room = min(
                                model.config.context_length - len(prompt_ids) - generated_count,
                                max_new_tokens - generated_count,
                            )
                            if len(result_ids) > room:
                                events.put({
                                    "kind": "tool_error",
                                    "call_id": call_id,
                                    "text": "completion budget leaves no room for tool result",
                                    "request": line,
                                })
                                return
                            events.put({
                                "kind": "tool_result",
                                "call_id": call_id,
                                "request": line,
                                "result": exact_call.result,
                            })
                            for character, result_id in zip(exact_call.result, result_ids):
                                events.put({
                                    "kind": "primary_token",
                                    "text": character,
                                    "source": "tool",
                                    "call_id": call_id,
                                })
                            generated_count += len(result_ids)
                            line += exact_call.result
                            pending = torch.cat(
                                (token, torch.tensor([result_ids], dtype=torch.long, device=device)),
                                dim=1,
                            )
                            continue
            pending = token

        for future in sidecar_futures:
            future.result()


def stream_inference_events(
    model: TinyMathGPT,
    tokenizer: CharacterTokenizer,
    question: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """Yield interleaved primary, tool, and sidecar events as inference progresses."""
    if not question.strip():
        raise ValueError("Enter a nonempty question")
    if max_new_tokens < 1 or temperature < 0:
        raise ValueError("Completion length must be positive and temperature nonnegative")
    unsupported = unsupported_characters(tokenizer, question)
    if unsupported:
        rendered = ", ".join(f"{char!r} (U+{ord(char):04X})" for char in unsupported)
        raise ValueError(f"Question contains characters outside this checkpoint tokenizer: {rendered}")
    if tokenizer.protocol != "reasoning_v3":
        raise ValueError("Interactive tool streaming currently requires a reasoning_v3 checkpoint")
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            _primary_generation(
                model,
                tokenizer,
                question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                events=events,
            )
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
            events.put({"kind": "generation_error", "text": str(exc)})
        finally:
            events.put({"kind": "done"})

    thread = threading.Thread(target=worker, name="tinymath-primary", daemon=True)
    thread.start()
    while True:
        event = events.get()
        if event["kind"] == "done":
            break
        yield event
    thread.join()
