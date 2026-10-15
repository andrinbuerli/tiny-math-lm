"""Build a reasoning_v3 bundle with parent and arithmetic-subtask rows."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.data.reasoning import (
    DELEGATION_POLICY,
    SUBTASK_SCHEMA_VERSION,
    annotate_row,
    distribution,
    generate_subtask_rows,
    verify_annotation,
)

_WORKER_TOKENIZER: CharacterTokenizer | None = None
_WORKER_MAX_TOKENS = 512
LOGGER = logging.getLogger(__name__)


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


def _sequence_length(row: dict[str, str], scratchpad: str) -> int:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Reasoning-v3 process worker has no tokenizer")
    prompt = _WORKER_TOKENIZER.encode(format_prompt(row["question"], protocol="reasoning_v3"), add_bos=True)
    completion = _WORKER_TOKENIZER.encode_completion(row["answer"], scratchpad=scratchpad)
    return len(prompt) + len(completion)


def _init_annotation_worker(tokenizer: CharacterTokenizer | None = None, max_tokens: int = 512) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_TOKENS
    _WORKER_TOKENIZER = tokenizer
    _WORKER_MAX_TOKENS = max_tokens


def _scan_row(
    payload: tuple[str, int, dict[str, str]],
) -> tuple[list[dict[str, str]], tuple[str, str, str] | None]:
    split, index, row = payload
    annotation = annotate_row(row, max_tokens=10**9, protocol="reasoning_v3")
    verify_annotation(row, annotation)
    subtasks = _subtasks(row, annotation, split, index, 10**9)
    training_text = None
    if split == "sft_train":
        training_text = (row["question"], row["answer"], str(annotation["scratchpad"]))
    return subtasks, training_text


def _annotate_row(payload: tuple[str, int, dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    split, index, row = payload
    try:
        annotation = annotate_row(
            row, max_tokens=_WORKER_MAX_TOKENS, length_fn=_sequence_length, protocol="reasoning_v3"
        )
        verify_annotation(row, annotation, length_fn=_sequence_length)
        subtasks = _subtasks(
            row, annotation, split, index, _WORKER_MAX_TOKENS, _sequence_length
        )
    except ValueError as exc:
        raise ValueError(f"{split}[{index}] {row.get('question', '')!r}: {exc}") from exc
    return annotation, subtasks


def _bounded_map(executor: ProcessPoolExecutor, function, rows, *, workers: int):
    """Map in input order while keeping only a small number of tasks in flight."""
    rows = iter(rows)
    pending = deque()
    limit = max(2, workers * 2)
    for _ in range(limit):
        try:
            pending.append(executor.submit(function, next(rows)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(function, next(rows)))
        except StopIteration:
            pass


def build_reasoning_v3_bundle(
    source: Path, destination: Path, *, max_tokens: int = 512, workers: int = 8,
    progress: Callable[[str, str, int], None] | None = None,
) -> dict[str, Any]:
    """Build off to the side and publish only a complete verified replacement."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and derived bundle directories must not overlap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / f".{destination.name}.build.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is already building this bundle") from None
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
        backup = None
        try:
            result = _build_reasoning_v3_bundle(
                source, staging, max_tokens=max_tokens, workers=workers, progress=progress,
            )
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.previous-{uuid.uuid4().hex[:12]}")
                destination.rename(backup)
            try:
                staging.rename(destination)
            except BaseException:
                if backup is not None:
                    backup.rename(destination)
                raise
            if backup is not None:
                LOGGER.info("Preserved previous bundle at %s", backup)
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _build_reasoning_v3_bundle(
    source: Path, destination: Path, *, max_tokens: int = 512, workers: int = 8,
    progress: Callable[[str, str, int], None] | None = None,
) -> dict[str, Any]:
    """Create combined parent/subtask splits, failing closed on any bad row."""
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
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    # Pass one verifies independent source rows in bounded parallel batches.
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_annotation_worker) as pool:
        for split, metadata in split_info.items():
            payloads = ((split, i, row) for i, row in enumerate(_rows(source / str(metadata["path"]))))
            for scanned, (subtasks, training_text) in enumerate(
                _bounded_map(pool, _scan_row, payloads, workers=workers), 1
            ):
                if scanned % 5000 == 0 or scanned == parent_counts[split]:
                    LOGGER.info("scan %s: %d/%d parents", split, scanned, parent_counts[split])
                    if progress is not None:
                        progress("scan", split, scanned)
                if training_text is not None:
                    for text in training_text:
                        chars.update(text)
                for subtask in subtasks:
                    task_splits[str(subtask["task_key"])].add(split)

    # Fit worker text only on tasks that remain eligible after checking every split.
    eligible_keys = {
        key for key, splits in task_splits.items()
        if "sft_train" in splits and not (splits - {"sft_train"})
    }
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_annotation_worker) as pool:
        payloads = (("sft_train", i, row) for i, row in enumerate(
            _rows(source / str(split_info["sft_train"]["path"]))
        ))
        for scanned, (subtasks, _) in enumerate(
            _bounded_map(pool, _scan_row, payloads, workers=workers), 1
        ):
            if scanned % 5000 == 0 or scanned == parent_counts["sft_train"]:
                LOGGER.info("vocabulary sft_train: %d/%d parents", scanned, parent_counts["sft_train"])
                if progress is not None:
                    progress("vocabulary", "sft_train", scanned)
            for subtask in subtasks:
                if str(subtask["task_key"]) in eligible_keys:
                    for field in ("question", "answer", "scratchpad"):
                        chars.update(str(subtask[field]))
    tokenizer = CharacterTokenizer(chars, protocol="reasoning_v3")

    destination.mkdir(parents=True, exist_ok=True)
    # Remove the layout emitted by early v3 builds so no stale special-purpose
    # subtask files remain beside the combined split files.
    shutil.rmtree(destination / "subtasks", ignore_errors=True)
    tokenizer_path = destination / "char_tokenizer.json"
    tokenizer.save(tokenizer_path)
    split_manifest: dict[str, dict[str, Any]] = {}
    parent_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    subtask_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
    parent_reduced: Counter[tuple[str, str]] = Counter()
    overlap_counts: Counter[str] = Counter()

    # Pass two annotates independent rows in parallel and writes results in source order.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_annotation_worker, initargs=(tokenizer, max_tokens)
    ) as pool:
        for split, metadata in split_info.items():
            split_path = destination / f"{split}.jsonl"
            split_digest = hashlib.sha256()
            parent_count = worker_count = 0
            source_rows = _rows(source / str(metadata["path"]))
            payloads = ((split, i, row) for i, row in enumerate(source_rows))
            with split_path.open("w", encoding="utf-8") as split_out:
                for (index, row), (annotation, subtasks) in zip(
                    enumerate(_rows(source / str(metadata["path"]))),
                    _bounded_map(pool, _annotate_row, payloads, workers=workers),
                    strict=True,
                ):
                    annotation["is_sub_task"] = False
                    for field in ("question", "answer", "scratchpad"):
                        _assert_known(tokenizer, str(row[field] if field != "scratchpad" else annotation[field]),
                                      f"{split}[{index}].{field}")
                    parent_line = _json_line(annotation)
                    split_out.write(parent_line)
                    split_digest.update(parent_line.encode("utf-8"))
                    parent_count += 1
                    if parent_count % 5000 == 0 or parent_count == parent_counts[split]:
                        LOGGER.info("write %s: %d/%d parents", split, parent_count, parent_counts[split])
                        if progress is not None:
                            progress("write", split, parent_count)
                    parent_lengths[(str(row["module"]), split)].append(int(annotation["retained_length"]))
                    if bool(annotation["reduced"]):
                        parent_reduced[(str(row["module"]), split)] += 1

                    for subtask in subtasks:
                        subtask["is_sub_task"] = True
                        for field in ("question", "answer", "scratchpad"):
                            _assert_known(tokenizer, str(subtask[field]),
                                          f"{split}[{index}].subtask.{field}")
                        key = str(subtask["task_key"])
                        overlaps = sorted(task_splits[key] - {split})
                        subtask["training_eligible"] = split == "sft_train" and not overlaps
                        subtask["overlapping_splits"] = overlaps
                        if overlaps:
                            overlap_counts[split] += 1
                        line = _json_line(subtask)
                        split_out.write(line)
                        split_digest.update(line.encode("utf-8"))
                        worker_count += 1
                        subtask_lengths[(str(subtask["module"]), split)].append(
                            int(subtask["retained_length"])
                        )
            split_manifest[split] = {
                "path": split_path.name, "examples": parent_count + worker_count,
                "parent_examples": parent_count, "subtask_examples": worker_count,
                "source_sha256": source_hashes[split], "sha256": _sha256(split_path),
                "annotation_sha256": split_digest.hexdigest(),
            }

    if sum(item["parent_examples"] for item in split_manifest.values()) != sum(parent_counts.values()):
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
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_operand_semantics": "atomic_rational",
        "delegation_policy": DELEGATION_POLICY,
        "splits": split_manifest,
        "subtask_training_eligible_tasks": len(eligible_keys),
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
        "subtask_schema_version": SUBTASK_SCHEMA_VERSION,
        "subtask_operand_semantics": "atomic_rational",
        "delegation_policy": DELEGATION_POLICY,
        "source_split_sha256": source_hashes, "splits": split_manifest,
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
