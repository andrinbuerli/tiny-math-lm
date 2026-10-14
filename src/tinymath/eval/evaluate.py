"""Greedy evaluation with identity-checked, resumable per-example predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from tinymath.checkpoint import (
    sha256_file,
    validate_checkpoint_compatibility,
)
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.completions import parse_reasoning_completion
from tinymath.data.prompts import format_prompt
from tinymath.data.splits import question_features
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.rewards.verify import verify_answer


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced_limit(examples: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    """Select a deterministic module-balanced subset for bounded smoke evaluations."""
    if limit is None or len(examples) <= limit:
        return examples
    grouped: dict[str, list[dict[str, str]]] = {}
    for example in examples:
        grouped.setdefault(example["module"], []).append(example)
    selected: list[dict[str, str]] = []
    offsets = {module: 0 for module in grouped}
    modules = sorted(grouped)
    while len(selected) < limit:
        made_progress = False
        for module in modules:
            index = offsets[module]
            if index < len(grouped[module]):
                selected.append(grouped[module][index])
                offsets[module] += 1
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
    return selected


def _checkpoint_provenance(checkpoint_path: Path, payload: dict[str, Any]) -> tuple[str, str | None]:
    """Read provenance from checkpoint fields, with run-directory fallbacks for older saves."""
    stage = checkpoint_path.resolve().parent.parent.name
    training_arm = payload.get("training_arm") or (
        stage if stage in {"sft", "grpo", "continued_sft"} else "unknown"
    )
    source_sft_checkpoint = payload.get("source_sft_checkpoint", payload.get("source_checkpoint"))
    if source_sft_checkpoint is None and training_arm == "continued_sft":
        config_path = checkpoint_path.resolve().parent / "resolved_config.json"
        try:
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            source_sft_checkpoint = saved_config.get("init_checkpoint")
        except (OSError, json.JSONDecodeError):
            pass
    if source_sft_checkpoint is not None:
        source_sft_checkpoint = str(Path(source_sft_checkpoint).resolve())
    return str(training_arm), source_sft_checkpoint


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "module", "question", "target", "prediction", "scratchpad", "correct",
        "format_valid", "capped", "output_tokens", "digit_width", "operator_count",
    ]
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in sorted(records, key=lambda item: item["row_index"]):
                writer.writerow({field: record.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _summarize(records: list[dict[str, Any]], prediction_path: str) -> dict[str, Any]:
    module_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"examples": 0, "correct": 0, "format_valid": 0, "capped": 0}
    )
    digit_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    expression_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        counts = module_counts[record["module"]]
        counts["examples"] += 1
        counts["correct"] += int(record["correct"])
        counts["format_valid"] += int(record["format_valid"])
        counts["capped"] += int(record["capped"])
        digit_counts[int(record["digit_width"])][0] += 1
        digit_counts[int(record["digit_width"])][1] += int(record["correct"])
        expression_counts[int(record["operator_count"])][0] += 1
        expression_counts[int(record["operator_count"])][1] += int(record["correct"])
    count = len(records)
    return {
        "examples": count,
        "scored_examples": count,
        "accuracy": sum(int(row["correct"]) for row in records) / count if count else None,
        "format_validity": sum(int(row["format_valid"]) for row in records) / count if count else None,
        "capped_completions": sum(int(row["capped"]) for row in records),
        "mean_output_tokens": sum(int(row["output_tokens"]) for row in records) / count if count else None,
        "module": {
            module: {
                **counts,
                "accuracy": counts["correct"] / counts["examples"] if counts["examples"] else 0.0,
                "format_validity": counts["format_valid"] / counts["examples"] if counts["examples"] else 0.0,
            }
            for module, counts in sorted(module_counts.items())
        },
        "by_digit_width": {
            str(key): {"examples": value[0], "accuracy": value[1] / value[0]}
            for key, value in sorted(digit_counts.items())
        },
        "by_operator_count": {
            str(key): {"examples": value[0], "accuracy": value[1] / value[0]}
            for key, value in sorted(expression_counts.items())
        },
        "predictions_path": prediction_path,
    }


def _identity_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    bundle_dir: Path,
    output_dir: Path,
    split_names: list[str],
    max_new_tokens: int,
    greedy: bool = True,
    temperature: float = 0.8,
    batch_size: int = 1,
    max_examples_per_split: int | None = None,
) -> dict[str, Any]:
    del batch_size  # autoregressive generation uses one unpadded prompt at a time
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not greedy and temperature <= 0:
        raise ValueError("Sampled evaluation requires a positive temperature")
    checkpoint_path = Path(checkpoint_path)
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_arm, source_sft_checkpoint = _checkpoint_provenance(checkpoint_path, checkpoint)
    tokenizer = CharacterTokenizer.load(bundle_dir / "char_tokenizer.json")
    model_config = GPTConfig.from_mapping(checkpoint["model_config"])
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    bundle_digest = sha256_file(manifest_path) if manifest_path.is_file() else _identity_digest(
        {name: sha256_file(bundle_dir / f"{name}.jsonl") for name in split_names}
    )
    annotation = manifest.get("annotation") or {}
    expected_compatibility = {
        "protocol": tokenizer.protocol,
        "annotation_sha256": annotation.get("annotation_sha256"),
        "tokenizer_sha256": tokenizer.sha256,
        "bundle_sha256": bundle_digest,
        "context_length": int(manifest.get("context_length", model_config.context_length)),
    }
    if int(model_config.context_length) != expected_compatibility["context_length"]:
        raise ValueError("Checkpoint context length does not match the evaluation bundle")
    validate_checkpoint_compatibility(
        checkpoint, expected_compatibility,
        allow_legacy_answer_only=(tokenizer.protocol == "answer_only"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMathGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, str]]] = {}
    split_digests: dict[str, str] = {}
    for split_name in split_names:
        source = bundle_dir / f"{split_name}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation split not found: {source}")
        split_digests[split_name] = sha256_file(source)
        split_rows[split_name] = _balanced_limit(_read_jsonl(source), max_examples_per_split)
    identity = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "bundle_sha256": bundle_digest,
        "split_sha256": split_digests,
        "protocol": tokenizer.protocol,
        "tokenizer_sha256": tokenizer.sha256,
        "annotation_sha256": annotation.get("annotation_sha256"),
        "context_length": model_config.context_length,
        "splits": list(split_names),
        "max_examples_per_split": max_examples_per_split,
        "decoding": {
            "strategy": "greedy" if greedy else "sampled",
            "temperature": 0.0 if greedy else temperature,
            "max_new_tokens": max_new_tokens,
        },
    }
    progress_path = output_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError("Existing evaluation progress belongs to a different checkpoint, split, or protocol")
        scored: dict[str, list[dict[str, Any]]] = progress.get("scored", {})
    else:
        scored = {name: [] for name in split_names}
    for split_name in split_names:
        scored.setdefault(split_name, [])

    summary_base = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_step": checkpoint.get("step", checkpoint.get("update")),
        "training_arm": training_arm,
        "source_sft_checkpoint": source_sft_checkpoint,
        "model_parameters": model.count_parameters(),
        "protocol": tokenizer.protocol,
        "annotation_sha256": annotation.get("annotation_sha256"),
        "bundle_sha256": bundle_digest,
        "tokenizer_sha256": tokenizer.sha256,
        "decoding": {
            "strategy": "greedy" if greedy else "sampled",
            "temperature": 0.0 if greedy else temperature,
            "max_new_tokens": max_new_tokens,
            "context_length": model_config.context_length,
            "per_prompt_cap": (
                "context_length - prompt_tokens" if tokenizer.protocol == "reasoning_v1"
                else "min(max_new_tokens, context_length - prompt_tokens)"
            ),
        },
    }

    def persist() -> dict[str, Any]:
        partial_splits = {}
        for name in split_names:
            prediction_name = f"{name}_predictions.csv"
            _atomic_predictions(output_dir / prediction_name, scored[name])
            partial_splits[name] = _summarize(scored[name], prediction_name)
        payload = {
            "identity": identity,
            "scored": scored,
            "splits": partial_splits,
        }
        _atomic_json(progress_path, payload)
        result = {**summary_base, "splits": partial_splits}
        _atomic_json(output_dir / "results.json", result)
        return result

    persist()
    for split_name in split_names:
        examples = split_rows[split_name]
        completed = {int(record["row_index"]) for record in scored[split_name]}
        for row_index, row in enumerate(examples):
            if row_index in completed:
                continue
            prompt_ids = tokenizer.encode(
                format_prompt(row["question"], protocol=tokenizer.protocol), add_bos=True
            )
            available = model_config.context_length - len(prompt_ids)
            if available < 1:
                raise ValueError("Evaluation prompt leaves no room for generated output")
            generation_cap = (
                available if tokenizer.protocol == "reasoning_v1"
                else min(max_new_tokens, available)
            )
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                generated = model.generate(
                    prompt,
                    max_new_tokens=generation_cap,
                    temperature=0.0 if greedy else temperature,
                    eos_id=tokenizer.eos_id,
                )[0, len(prompt_ids) :].tolist()
            saw_eos = tokenizer.eos_id in generated
            if saw_eos:
                generated = generated[: generated.index(tokenizer.eos_id) + 1]
            capped = not saw_eos and len(generated) >= generation_cap
            prediction = tokenizer.decode(generated, skip_special=True).strip()
            scratchpad = ""
            if tokenizer.protocol == "reasoning_v1":
                try:
                    parsed = parse_reasoning_completion(generated, tokenizer)
                    scratchpad = parsed.scratchpad
                    correct, valid = verify_answer(
                        parsed.answer, row["answer"], question=row["question"]
                    )
                except ValueError:
                    correct, valid = False, False
            else:
                correct, valid = verify_answer(prediction, row["answer"], question=row["question"])
            if capped:
                correct, valid = False, False
            digit_width, operator_count = question_features(row["question"])
            scored[split_name].append(
                {
                    "row_index": row_index,
                    "module": row["module"], "question": row["question"],
                    "target": row["answer"], "prediction": prediction,
                    "scratchpad": scratchpad, "correct": int(correct),
                    "format_valid": int(valid), "capped": int(capped),
                    "output_tokens": len(generated), "digit_width": digit_width,
                    "operator_count": operator_count,
                }
            )
            scored[split_name].sort(key=lambda item: item["row_index"])
            persist()
    return persist()
