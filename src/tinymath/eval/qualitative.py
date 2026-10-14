"""Deterministic qualitative examples for experiment inspection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.completions import parse_reasoning_completion
from tinymath.data.prompts import format_prompt
from tinymath.data.trace_explanations import transcribe_scratchpad
from tinymath.rewards.verify import verify_answer


def _selected_rows(
    rows: Sequence[dict[str, str]], *, split: str, seed: int, per_module: int
) -> list[dict[str, str]]:
    by_module: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_module.setdefault(row["module"], []).append(row)
    selected: list[dict[str, str]] = []
    for module, module_rows in sorted(by_module.items()):
        ranked = sorted(
            module_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{split}\0{module}\0{row['question']}".encode()
            ).digest(),
        )
        selected.extend(ranked[:per_module])
    return selected


def _predicted_scratchpad(
    generated: Sequence[int], tokenizer: CharacterTokenizer
) -> str:
    """Return generated scratchpad text for inspection, even when parsing fails."""
    if tokenizer.protocol not in {"reasoning_v1", "reasoning_v2"}:
        return ""
    try:
        start = generated.index(tokenizer.scp_id)
    except ValueError:
        return ""
    try:
        end = generated.index(tokenizer.scp_end_id, start + 1)
    except ValueError:
        end = next(
            (index for index in range(start + 1, len(generated)) if generated[index] == tokenizer.eos_id),
            len(generated),
        )
    return tokenizer.decode(generated[start + 1 : end], skip_special=True).strip()


@torch.no_grad()
def build_qualitative_table(
    *,
    model: torch.nn.Module,
    tokenizer: CharacterTokenizer,
    split_rows: Mapping[str, Sequence[dict[str, str]]],
    training_arm: str,
    seed: int,
    max_new_tokens: int = 128,
    examples_per_module: int = 2,
) -> tuple[Any, int, dict[str, float | None]]:
    """Build a deterministic W&B table with generated train/validation/test answers."""
    if examples_per_module < 1 or max_new_tokens < 1:
        raise ValueError("Qualitative sample and generation limits must be positive")
    device = next(model.parameters()).device
    context_length = int(model.config.context_length)
    was_training = model.training
    model.eval()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    rows: list[list[Any]] = []
    split_correct = {split: [0, 0] for split in ("train", "validation", "test")}
    try:
        for split in ("train", "validation", "test"):
            examples = _selected_rows(
                split_rows.get(split, ()), split=split, seed=seed, per_module=examples_per_module
            )
            for example in examples:
                prompt = torch.tensor(
                    [
                        tokenizer.encode(
                            format_prompt(example["question"], protocol=tokenizer.protocol),
                            add_bos=True,
                        )
                    ],
                    dtype=torch.long,
                    device=device,
                )
                available = context_length - prompt.shape[1]
                if available < 1:
                    raise ValueError(f"Qualitative {split} prompt leaves no room for an answer")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    generated = model.generate(
                        prompt,
                        max_new_tokens=(
                            available
                            if tokenizer.protocol in {"reasoning_v1", "reasoning_v2"}
                            else min(max_new_tokens, available)
                        ),
                        temperature=0.0,
                        eos_id=tokenizer.eos_id,
                    )[0, prompt.shape[1] :].tolist()
                if tokenizer.eos_id in generated:
                    generated = generated[: generated.index(tokenizer.eos_id) + 1]
                completion = tokenizer.decode(generated, skip_special=True).strip()
                predicted_scratchpad = _predicted_scratchpad(generated, tokenizer)
                transcribed_scratchpad = ""
                trace_structure_valid: bool | None = None
                if tokenizer.protocol in {"reasoning_v1", "reasoning_v2"}:
                    try:
                        parsed = parse_reasoning_completion(generated, tokenizer)
                        predicted_scratchpad = parsed.scratchpad
                        explanation = transcribe_scratchpad(
                            parsed.scratchpad, version=tokenizer.protocol
                        )
                        if not explanation:
                            raise ValueError("Scratchpad has no transcribable reasoning steps")
                        transcribed_scratchpad = "\n".join(explanation)
                        prediction = parsed.answer
                        correct, valid = verify_answer(
                            prediction, example["answer"], question=example["question"]
                        )
                        trace_structure_valid = True
                    except ValueError:
                        prediction = ""
                        correct, valid = False, False
                        trace_structure_valid = False
                else:
                    prediction = completion
                    correct, valid = verify_answer(
                        prediction, example["answer"], question=example["question"]
                    )
                    trace_structure_valid = None
                split_correct[split][0] += int(correct)
                split_correct[split][1] += 1
                rows.append(
                    [
                        training_arm,
                        seed,
                        split,
                        example["module"],
                        example["question"],
                        example["answer"],
                        prediction,
                        completion,
                        predicted_scratchpad,
                        transcribed_scratchpad,
                        trace_structure_valid,
                        correct,
                        valid,
                        len(generated),
                        int(
                            len(generated) >= available
                            and (not generated or generated[-1] != tokenizer.eos_id)
                        ),
                    ]
                )
    finally:
        model.train(was_training)

    import wandb

    table = wandb.Table(
        columns=[
            "training_arm",
            "seed",
            "split",
            "module",
            "question",
            "target",
            "prediction",
            "completion",
            "predicted_scratchpad",
            "transcribed_scratchpad",
            "trace_structure_valid",
            "correct",
            "format_valid",
            "output_tokens",
            "capped",
        ],
        data=rows,
    )
    accuracy = {
        split: correct / total if total else None
        for split, (correct, total) in split_correct.items()
    }
    return table, len(rows), accuracy


def log_qualitative_examples(
    *,
    model: torch.nn.Module,
    tokenizer: CharacterTokenizer,
    split_rows: Mapping[str, Sequence[dict[str, str]]],
    run: Any,
    training_arm: str,
    seed: int,
    step: int,
    max_new_tokens: int = 128,
    examples_per_module: int = 2,
) -> int:
    """Log deterministic train/validation/test prompts with generated answers."""
    table, count, accuracy = build_qualitative_table(
        model=model,
        tokenizer=tokenizer,
        split_rows=split_rows,
        training_arm=training_arm,
        seed=seed,
        max_new_tokens=max_new_tokens,
        examples_per_module=examples_per_module,
    )
    # The resumed W&B history may already be past the sampled checkpoint step.
    # Keep optimizer_step as a column and let W&B assign the next history step.
    run.log(
        {
            "qualitative/examples": table,
            "optimizer_step": step,
            **{
                f"{'val' if split == 'validation' else split}/greedy_accuracy": value
                for split, value in accuracy.items()
            },
        }
    )
    summary = getattr(run, "summary", None)
    if summary is not None:
        summary["qualitative/example_count"] = count
        summary["qualitative/splits"] = ["train", "validation", "test"]
    return count
