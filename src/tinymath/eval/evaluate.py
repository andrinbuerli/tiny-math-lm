"""Greedy evaluation and per-example provenance output."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.data.splits import question_features
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.rewards.verify import verify_answer


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    bundle_dir: Path,
    output_dir: Path,
    split_names: list[str],
    max_new_tokens: int,
    batch_size: int = 1,
    max_examples_per_split: int | None = None,
) -> dict[str, Any]:
    del batch_size  # autoregressive generation uses one unpadded prompt at a time
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = CharacterTokenizer.load(bundle_dir / "char_tokenizer.json")
    if checkpoint.get("tokenizer_sha256") != tokenizer.sha256:
        raise ValueError("Checkpoint tokenizer does not match the evaluation data bundle")
    model_config = GPTConfig.from_mapping(checkpoint["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMathGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        model.to(dtype=torch.bfloat16)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for split_name in split_names:
        source = bundle_dir / f"{split_name}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation split not found: {source}")
        examples = _read_jsonl(source)
        if max_examples_per_split is not None:
            examples = examples[:max_examples_per_split]
        module_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"examples": 0, "correct": 0, "format_valid": 0}
        )
        digit_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        expression_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        correct_total = valid_total = output_tokens = 0
        prediction_path = output_dir / f"{split_name}_predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "module", "question", "target", "prediction", "correct", "format_valid",
                    "output_tokens", "digit_width", "operator_count",
                ],
            )
            writer.writeheader()
            for row in examples:
                prompt = torch.tensor(
                    [tokenizer.encode(format_prompt(row["question"]), add_bos=True)],
                    dtype=torch.long,
                    device=device,
                )
                available = model_config.context_length - prompt.shape[1]
                if available < 1:
                    raise ValueError("Evaluation prompt leaves no room for generated output")
                generated = model.generate(
                    prompt,
                    max_new_tokens=min(max_new_tokens, available),
                    temperature=0.0,
                    eos_id=tokenizer.eos_id,
                )[0, prompt.shape[1] :].tolist()
                if tokenizer.eos_id in generated:
                    generated = generated[: generated.index(tokenizer.eos_id) + 1]
                prediction = tokenizer.decode(generated, skip_special=True).strip()
                correct, valid = verify_answer(prediction, row["answer"])
                digit_width, operator_count = question_features(row["question"])
                counts = module_counts[row["module"]]
                counts["examples"] += 1
                counts["correct"] += int(correct)
                counts["format_valid"] += int(valid)
                digit_counts[digit_width][0] += 1
                digit_counts[digit_width][1] += int(correct)
                expression_counts[operator_count][0] += 1
                expression_counts[operator_count][1] += int(correct)
                correct_total += int(correct)
                valid_total += int(valid)
                output_tokens += len(generated)
                writer.writerow(
                    {
                        "module": row["module"], "question": row["question"], "target": row["answer"],
                        "prediction": prediction, "correct": int(correct), "format_valid": int(valid),
                        "output_tokens": len(generated), "digit_width": digit_width,
                        "operator_count": operator_count,
                    }
                )
        results[split_name] = {
            "examples": len(examples),
            "accuracy": correct_total / len(examples) if examples else 0.0,
            "format_validity": valid_total / len(examples) if examples else 0.0,
            "mean_output_tokens": output_tokens / len(examples) if examples else 0.0,
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
            "predictions_path": prediction_path.name,
        }
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": checkpoint.get("step", checkpoint.get("update")),
        "source_sft_checkpoint": checkpoint.get("source_sft_checkpoint"),
        "tokenizer_sha256": tokenizer.sha256,
        "splits": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
