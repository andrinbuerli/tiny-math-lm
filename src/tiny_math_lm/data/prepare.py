"""Build deterministic split files, a frozen tokenizer and a provenance manifest."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from tiny_math_lm.config import WORKSPACE
from tiny_math_lm.data.char_tokenizer import CharacterTokenizer
from tiny_math_lm.data.load import DatasetInspection
from tiny_math_lm.data.prompts import format_completion, format_prompt
from tiny_math_lm.data.splits import DataPools, split_records


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
        "max": ordered[-1],
    }


def make_tokenizer(pools: DataPools) -> CharacterTokenizer:
    """Fit on selected training pools and the fixed output framing only."""
    texts = ["Q: \nA: "]
    for split in ("sft_train", "sft_validation", "rlvr"):
        texts.extend(
            format_prompt(row["question"]) + format_completion(row["answer"])
            for row in pools.splits[split]
        )
    tokenizer = CharacterTokenizer.from_texts(texts)
    for rows in pools.splits.values():
        for row in rows:
            # Unknown test characters are explicit data-preparation errors.
            tokenizer.encode(format_prompt(row["question"]) + format_completion(row["answer"]))
    return tokenizer


def write_data_bundle(
    pools: DataPools,
    inspection: DatasetInspection,
    *,
    output_dir: str | Path,
    tokenizer: CharacterTokenizer,
    seed: int,
    sft_examples: int,
    context_length: int,
) -> dict[str, Any]:
    """Persist all disjoint pools and return the split manifest."""
    destination = Path(output_dir).resolve()
    if WORKSPACE not in destination.parents:
        raise ValueError(f"Prepared data must stay inside {WORKSPACE}")
    destination.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    selected_counts: dict[str, dict[str, dict[str, int]]] = {}
    train_lengths: list[int] = []
    module_token_counts: Counter[str] = Counter()

    for split_name, rows in pools.splits.items():
        path = destination / f"{split_name}.jsonl"
        counts: Counter[tuple[str, str]] = Counter()
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                prompt = format_prompt(row["question"])
                completion = format_completion(row["answer"])
                length = len(tokenizer.encode(prompt, add_bos=True)) + len(
                    tokenizer.encode(completion, add_eos=True)
                )
                if length > context_length:
                    raise ValueError(f"Overlength row reached writer for {split_name}: {length}")
                handle.write(_json_line(row))
                counts[(row["module"], row["category"])] += 1
                if split_name in ("sft_train", "sft_validation", "rlvr"):
                    train_lengths.append(length)
                    module_token_counts[row["module"]] += length
        split_files[split_name] = {"path": path.name, "examples": len(rows), "sha256": _sha256(path)}
        selected_counts[split_name] = {
            module: {
                category: counts[(module, category)]
                for category in sorted({row["category"] for row in rows})
            }
            for module in sorted({row["module"] for row in rows})
        }

    tokenizer_path = destination / "char_tokenizer.json"
    tokenizer.save(tokenizer_path)
    samples = {
        split_name: rows[: min(3, len(rows))]
        for split_name, rows in pools.splits.items()
    }
    samples_path = destination / "inspected_samples.json"
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "dataset_id": inspection.dataset_id,
        "dataset_config": inspection.config,
        "dataset_split": inspection.split,
        "dataset_revision": inspection.revision,
        "dataset_schema": inspection.features,
        "source_split_examples": inspection.split_examples,
        "source_split_bytes": inspection.split_bytes,
        "selected_modules": list(pools.raw_counts),
        "source_category_counts": pools.raw_counts,
        "selected_pool_category_counts": selected_counts,
        "excluded_overlength_counts": pools.excluded_overlength,
        "unique_selected_source_questions": sum(split_file["examples"] for split_file in split_files.values()),
        "duplicate_questions_removed": pools.duplicate_questions,
        "seed": seed,
        "sft_train_examples": sft_examples,
        "context_length": context_length,
        "prompt_format": "Q: {original_question}\\nA: ",
        "answer_format": "original answer followed by EOS",
        "split_rules": {
            "train_categories": ["train-easy", "train-medium", "train-hard"],
            "test_category": "interpolate",
            "ood_source_by_module": pools.ood_source_by_module,
            "deduplication": "NFC and whitespace-collapse question key; preserve original text; official evaluation category wins",
            "module_balance": "equal requested example counts across the five modules, remainder assigned in configured order",
            "number_length_ood": "rows above module interpolate p95 max digit-run width while operator count stays at or below p95; use official extrapolate rows where available, otherwise reserve deterministic train-category rows",
            "expression_length_ood": "rows above module interpolate p95 operator count while max digit-run width stays at or below p95; use official extrapolate rows where available, otherwise reserve deterministic train-category rows",
            "extrapolate_other": "all remaining official extrapolate examples; it is disjoint from the two controlled OOD subsets",
            "overlength": "exclude examples whose prompt+completion+special-token length exceeds context; never truncate",
        },
        "ood_thresholds": pools.thresholds,
        "splits": split_files,
        "tokenizer": {
            "version": 1,
            "strict_single_unicode_codepoint": True,
            "special_tokens": ["PAD", "BOS", "EOS", "UNK"],
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer.sha256,
            "path": tokenizer_path.name,
        },
        "token_length_distribution_training_pools": _summary(train_lengths),
        "token_counts_by_module_training_pools": dict(module_token_counts),
        "inspected_samples_path": samples_path.name,
    }
    hashable_manifest = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["split_manifest_sha256"] = hashlib.sha256(hashable_manifest.encode("utf-8")).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def prepare_bundle(
    records: list[dict[str, str]],
    inspection: DatasetInspection,
    *,
    output_dir: str | Path,
    seed: int,
    sft_examples: int,
    validation_fraction: float,
    rlvr_fraction: float,
    context_length: int,
    ood_examples_per_module: int = 1000,
) -> tuple[DataPools, CharacterTokenizer, dict[str, Any]]:
    pools = split_records(
        records,
        sft_examples=sft_examples,
        validation_fraction=validation_fraction,
        rlvr_fraction=rlvr_fraction,
        seed=seed,
        context_length=context_length,
        ood_examples_per_module=ood_examples_per_module,
    )
    tokenizer = make_tokenizer(pools)
    manifest = write_data_bundle(
        pools,
        inspection,
        output_dir=output_dir,
        tokenizer=tokenizer,
        seed=seed,
        sft_examples=sft_examples,
        context_length=context_length,
    )
    return pools, tokenizer, manifest
