"""Build a separate reasoning_v3 parent and arithmetic-subtask data bundle."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.data.reasoning import (
    annotate_row,
    distribution,
    generate_subtask_rows,
    verify_annotation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _subtasks(row: dict[str, str], annotation: dict[str, object], split: str, index: int,
              limit: int, length_fn=None) -> list[dict[str, object]]:
    options = {} if length_fn is None else {"length_fn": length_fn}
    return generate_subtask_rows(row, annotation, source_split=split,
                                  parent_row_index=index, max_tokens=limit, **options)


def build_reasoning_v3_bundle(
    source: Path, destination: Path, *, max_tokens: int = 512
) -> dict[str, Any]:
    """Create parent and standalone subtask splits, failing closed on any bad row."""
    manifest_path = source / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_info = source_manifest.get("splits")
    if not isinstance(split_info, dict) or not split_info:
        raise ValueError("Source manifest has no split inventory")
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError("The derived v3 bundle must not overwrite or nest inside its source")

    expected_pool_count = source_manifest.get("token_length_distribution_training_pools", {}).get("count")
    parent_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for split, metadata in split_info.items():
        path = source / str(metadata["path"])
        count = sum(1 for _ in _rows(path))
        parent_counts[split] = count
        source_hashes[split] = _sha256(path)
        if "examples" in metadata and count != int(metadata["examples"]):
            raise ValueError(f"Source manifest count mismatch for {split}: {count}")
    if expected_pool_count is not None:
        actual = sum(parent_counts.get(name, 0) for name in ("sft_train", "sft_validation", "rlvr"))
        if actual != int(expected_pool_count):
            raise ValueError(f"Source training-pool count mismatch: {actual} != {expected_pool_count}")

    task_splits: dict[str, set[str]] = defaultdict(set)
    chars: set[str] = set(
        "0123456789+-*/(),.:;=@^> <" + "\n" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    # Pass one verifies source rows and discovers where each canonical subtask occurs.
    for split, metadata in split_info.items():
        for index, row in enumerate(_rows(source / str(metadata["path"]))):
            annotation = annotate_row(row, max_tokens=10**9, protocol="reasoning_v3")
            verify_annotation(row, annotation)
            if split == "sft_train":
                chars.update(row["question"])
                chars.update(row["answer"])
                chars.update(str(annotation["scratchpad"]))
            for subtask in _subtasks(row, annotation, split, index, 10**9):
                task_splits[str(subtask["task_key"])].add(split)

    # Fit worker text only on tasks that remain eligible after checking every split.
    eligible_keys = {
        key for key, splits in task_splits.items()
        if "sft_train" in splits and not (splits - {"sft_train"})
    }
    for index, row in enumerate(_rows(source / str(split_info["sft_train"]["path"]))):
        annotation = annotate_row(row, max_tokens=10**9, protocol="reasoning_v3")
        for subtask in _subtasks(row, annotation, "sft_train", index, 10**9):
            if str(subtask["task_key"]) in eligible_keys:
                chars.update(str(subtask[field]) for field in ("question", "answer", "scratchpad"))
    tokenizer = CharacterTokenizer(chars, protocol="reasoning_v3")

    def sequence_length(row: dict[str, str], scratchpad: str) -> int:
        prompt = tokenizer.encode(format_prompt(row["question"], protocol="reasoning_v3"), add_bos=True)
        completion = tokenizer.encode_completion(row["answer"], scratchpad=scratchpad)
        return len(prompt) + len(completion)

    destination.mkdir(parents=True, exist_ok=True)
    subtask_dir = destination / "subtasks"
    subtask_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = destination / "char_tokenizer.json"
    tokenizer.save(tokenizer_path)
    parent_manifest: dict[str, dict[str, Any]] = {}
    subtask_manifest: dict[str, dict[str, Any]] = {}
    parent_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    subtask_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    parent_reduced: Counter[tuple[str, str]] = Counter()
    overlap_counts: Counter[str] = Counter()

    # Pass two writes immutable-source parent rows and their linked worker samples.
    for split, metadata in split_info.items():
        parent_path = destination / f"{split}.jsonl"
        worker_path = subtask_dir / f"{split}.jsonl"
        parent_digest, worker_digest = hashlib.sha256(), hashlib.sha256()
        parent_count = worker_count = 0
        with parent_path.open("w", encoding="utf-8") as parent_out, worker_path.open(
            "w", encoding="utf-8"
        ) as worker_out:
            for index, row in enumerate(_rows(source / str(metadata["path"]))):
                annotation = annotate_row(
                    row, max_tokens=max_tokens, length_fn=sequence_length, protocol="reasoning_v3"
                )
                verify_annotation(row, annotation)
                for field in ("question", "answer", "scratchpad"):
                    _assert_known(tokenizer, str(row[field] if field != "scratchpad" else annotation[field]),
                                  f"{split}[{index}].{field}")
                parent_line = _json_line(annotation)
                parent_out.write(parent_line)
                parent_digest.update(parent_line.encode("utf-8"))
                parent_count += 1
                parent_lengths[(str(row["module"]), split)].append(int(annotation["retained_length"]))
                if bool(annotation["reduced"]):
                    parent_reduced[(str(row["module"]), split)] += 1

                subtasks = _subtasks(row, annotation, split, index, max_tokens, sequence_length)
                for subtask in subtasks:
                    for field in ("question", "answer", "scratchpad"):
                        _assert_known(tokenizer, str(subtask[field]),
                                      f"subtasks/{split}[{index}].{field}")
                    key = str(subtask["task_key"])
                    overlaps = sorted(task_splits[key] - {split})
                    eligible = split == "sft_train" and not overlaps
                    subtask["training_eligible"] = eligible
                    subtask["overlapping_splits"] = overlaps
                    if overlaps:
                        overlap_counts[split] += 1
                    line = _json_line(subtask)
                    worker_out.write(line)
                    worker_digest.update(line.encode("utf-8"))
                    worker_count += 1
                    subtask_lengths[(str(subtask["module"]), split)].append(
                        int(subtask["retained_length"])
                    )
        parent_manifest[split] = {
            "path": parent_path.name, "examples": parent_count,
            "source_sha256": source_hashes[split], "sha256": _sha256(parent_path),
            "annotation_sha256": parent_digest.hexdigest(),
        }
        subtask_manifest[split] = {
            "path": str(worker_path.relative_to(destination)), "examples": worker_count,
            "sha256": _sha256(worker_path), "annotation_sha256": worker_digest.hexdigest(),
        }

    if sum(item["examples"] for item in parent_manifest.values()) != sum(parent_counts.values()):
        raise AssertionError("Annotated parent count differs from source split inventory")
    tokenizer_stats = {
        f"parent/{module}/{split}": distribution(lengths)
        for (module, split), lengths in sorted(parent_lengths.items())
    }
    tokenizer_stats.update({
        f"subtask/{module}/{split}": distribution(lengths)
        for (module, split), lengths in sorted(subtask_lengths.items())
    })
    training_parent_count = sum(parent_counts.get(name, 0) for name in ("sft_train", "sft_validation", "rlvr"))
    training_parent_lengths = [
        length for (module, split), values in parent_lengths.items()
        if split in {"sft_train", "sft_validation", "rlvr"} for length in values
    ]
    training_token_counts: Counter[str] = Counter()
    for (module, split), values in parent_lengths.items():
        if split in {"sft_train", "sft_validation", "rlvr"}:
            training_token_counts[module] += sum(values)
    derived = dict(source_manifest)
    derived.update({
        "experiment_identity": "reasoning_v3_arithmetic_subtasks",
        "source_bundle": str(source.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "context_length": max_tokens,
        "prompt_format": "Q: {original_question}\\n",
        "completion_format": "<scp>\\n{scratchpad}\\n</scp>\\nA: {original_answer}<EOS>",
        "trace_protocol": "reasoning_v3",
        "splits": parent_manifest,
        "subtask_splits": subtask_manifest,
        "subtask_training_eligible": len(eligible_keys),
        "subtask_overlapping_occurrences_by_split": dict(overlap_counts),
        "token_counts_by_module_training_pools": dict(sorted(training_token_counts.items())),
        "token_length_statistics": tokenizer_stats,
        "token_length_distribution_training_pools": distribution(training_parent_lengths),
        "tokenizer": {
            "version": 4, "protocol": "reasoning_v3", "strict_single_unicode_codepoint": True,
            "special_tokens": list(tokenizer.special_tokens), "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer.sha256, "path": tokenizer_path.name,
        },
    })
    annotation = {
        "protocol": "reasoning_v3", "max_tokens": max_tokens,
        "source_split_sha256": source_hashes, "splits": parent_manifest,
        "subtask_splits": subtask_manifest,
        "tokenizer_sha256": tokenizer.sha256,
        "training_parent_rows": training_parent_count,
        "eligible_subtask_tasks": len(eligible_keys),
        "subtask_overlapping_occurrences_by_split": dict(overlap_counts),
        "module_split_statistics": tokenizer_stats,
    }
    annotation["annotation_sha256"] = hashlib.sha256(
        json.dumps(annotation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    derived["annotation"] = annotation
    (destination / "manifest.json").write_text(
        json.dumps(derived, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return derived


def _assert_known(tokenizer: CharacterTokenizer, text: str, context: str) -> None:
    ids = tokenizer.encode(text, allow_unk=True)
    if tokenizer.unk_id in ids:
        unknown = next(char for char, token in zip(text, ids) if token == tokenizer.unk_id)
        raise ValueError(f"Unknown tokenizer character in {context}: U+{ord(unknown):04X} {unknown!r}")
