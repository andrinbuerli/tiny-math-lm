"""Versioned native-token cache for Experiment 16."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tinymath.data.hf_reasoning import (
    FORMAT_VERSION,
    encode_completion,
    sha256_file,
    validate_source_bundle,
)
from tinymath.data.load import SELECTED_MODULES

SPLIT_FILES = {
    "train": "sft_train.jsonl",
    "validation": "sft_validation.jsonl",
    "rlvr": "rlvr.jsonl",
    "test": "test.jsonl",
    "number_length_ood": "number_length_ood.jsonl",
    "expression_length_ood": "expression_length_ood.jsonl",
}


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def native_tokenizer_identity(tokenizer: Any) -> tuple[str, dict[str, Any]]:
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    if len(think_ids) != 1 or think_ids != [tokenizer.convert_tokens_to_ids("<think>")]:
        raise ValueError("The pinned Qwen tokenizer does not provide one existing <think> token")
    if len(end_think_ids) != 1 or end_think_ids != [tokenizer.convert_tokens_to_ids("</think>")]:
        raise ValueError("The pinned Qwen tokenizer does not provide one existing </think> token")
    identity = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "think_token_id": think_ids[0],
        "end_think_token_id": end_think_ids[0],
        "special_tokens_map": tokenizer.special_tokens_map,
        "backend_tokenizer_sha256": hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode("utf-8")
        ).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return digest, identity


def _row_identity(split: str, index: int, row: dict[str, Any]) -> str:
    payload = {
        "split": split,
        "index": index,
        "module": row.get("module"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "is_sub_task": bool(row.get("is_sub_task", False)),
        "task_key": row.get("task_key"),
        "training_eligible": row.get("training_eligible"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _selected(split: str, rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    if split == "train":
        def training_rows():
            for row in rows:
                if row.get("module") not in SELECTED_MODULES:
                    raise ValueError("SFT source row is outside the required five-module scope")
                if row.get("is_sub_task", False):
                    if row.get("training_eligible") is not True:
                        continue
                    if row.get("overlapping_splits"):
                        raise ValueError("Eligible standalone subtask has a cross-pool overlap")
                yield row

        return training_rows()

    def parent_rows():
        for row in rows:
            if row.get("module") not in SELECTED_MODULES:
                raise ValueError("Evaluation source row is outside the required five-module scope")
            if not row.get("is_sub_task", False):
                yield row

    # Validation and fixed or generation pools are parent-only.
    return parent_rows()


def _write_split(
    *,
    split: str,
    source_path: Path,
    destination: Path,
    tokenizer: Any,
    context_length: int,
    completion_cap: int,
) -> dict[str, Any]:
    lengths: list[int] = []
    completion_lengths: list[int] = []
    row_digest = hashlib.sha256()
    fixed_identities: list[str] = []
    ids_path = destination / f"{split}.ids.u32le"
    offsets_path = destination / f"{split}.offsets.u64le"
    prompt_path = destination / f"{split}.prompt_lengths.u32le"
    metadata_path = destination / f"{split}.rows.jsonl"
    offset = 0
    with (
        ids_path.open("wb") as ids_file,
        offsets_path.open("wb") as offsets_file,
        prompt_path.open("wb") as prompt_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        offsets_file.write(struct.pack("<Q", 0))
        for index, row in enumerate(_selected(split, iter_rows(source_path))):
            ids, _labels, prompt_length = encode_completion(
                tokenizer,
                row,
                context_length=context_length,
                completion_cap=completion_cap,
            )
            completion_length = len(ids) - prompt_length
            ids_file.write(struct.pack(f"<{len(ids)}I", *ids))
            offset += len(ids)
            offsets_file.write(struct.pack("<Q", offset))
            prompt_file.write(struct.pack("<I", prompt_length))
            metadata = {
                **row,
                "cache_row_identity": _row_identity(split, index, row),
                "prompt_tokens": prompt_length,
                "completion_tokens": completion_length,
            }
            metadata_file.write(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            row_digest.update(metadata["cache_row_identity"].encode("ascii"))
            row_digest.update(b"\n")
            lengths.append(len(ids))
            completion_lengths.append(completion_length)
            fixed_identities.append(
                json.dumps(
                    [row.get("module", ""), row.get("question", ""), row.get("answer", "")],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    if not lengths:
        raise ValueError(f"Native-token split {split} is empty")
    fixed_identity_digest = hashlib.sha256()
    for identity in sorted(fixed_identities):
        fixed_identity_digest.update(identity.encode("utf-8"))
        fixed_identity_digest.update(b"\n")
    return {
        "rows": len(lengths),
        "row_identity_sha256": row_digest.hexdigest(),
        "fixed_question_identity_sha256": fixed_identity_digest.hexdigest(),
        "fixed_question_identity_count": len(fixed_identities),
        "token_lengths": {
            "min": int(min(lengths)),
            "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)),
            "p99": float(np.quantile(lengths, 0.99)),
            "max": int(max(lengths)),
            "mean": float(np.mean(lengths)),
        },
        "completion_tokens": {
            "min": int(min(completion_lengths)),
            "median": float(np.median(completion_lengths)),
            "p95": float(np.quantile(completion_lengths, 0.95)),
            "p99": float(np.quantile(completion_lengths, 0.99)),
            "max": int(max(completion_lengths)),
        },
        "files": {
            path.name: sha256_file(path)
            for path in (ids_path, offsets_path, prompt_path, metadata_path)
        },
    }


def prepare_hf_token_cache(
    *,
    bundle_dir: str | Path,
    cache_root: str | Path,
    tokenizer: Any,
    tokenizer_revision: str,
    config_sha256: str,
    expected_manifest_sha256: str,
    expected_eval_identities: dict[str, str],
    context_length: int = 512,
    completion_cap: int = 512,
) -> tuple[Path, dict[str, Any]]:
    """Tokenize all complete eligible SFT rows and parent-only evaluation pools."""
    bundle = Path(bundle_dir).resolve()
    source_manifest = validate_source_bundle(
        bundle, expected_manifest_sha256=expected_manifest_sha256
    )
    tokenizer_sha256, tokenizer_identity = native_tokenizer_identity(tokenizer)
    formatting_payload = {
        "version": FORMAT_VERSION,
        "prompt": "Q: {original question}\\n",
        "completion": "<think>\\n{scratchpad}\\n</think>\\nA: {answer}{native EOS}",
        "thinking_ids": [
            tokenizer_identity["think_token_id"], tokenizer_identity["end_think_token_id"]
        ],
        "context_length": context_length,
        "completion_cap": completion_cap,
    }
    formatting_sha256 = hashlib.sha256(
        json.dumps(formatting_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = (
        Path(cache_root).resolve()
        / f"exp16_qwen3_base_reasoning_v3-{tokenizer_revision[:12]}"
        / f"{tokenizer_sha256[:12]}-{formatting_sha256[:12]}"
    )
    expected = {
        "format_version": FORMAT_VERSION,
        "formatting_sha256": formatting_sha256,
        "source_bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "source_split_manifest_sha256": source_manifest["split_manifest_sha256"],
        "dataset_revision": source_manifest["dataset_revision"],
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_identity": tokenizer_identity,
        "model_config_sha256": config_sha256,
        "context_length": context_length,
        "completion_cap": completion_cap,
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            for split, info in cached["splits"].items():
                for filename, digest in info["files"].items():
                    if sha256_file(destination / filename) != digest:
                        break
                else:
                    continue
                break
            else:
                return destination, cached
        raise ValueError(f"Existing native token cache has incompatible or corrupt identity: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        split_results = {}
        for split, filename in SPLIT_FILES.items():
            split_results[split] = _write_split(
                split=split,
                source_path=bundle / filename,
                destination=staging,
                tokenizer=tokenizer,
                context_length=context_length,
                completion_cap=completion_cap,
            )
        for split, expected_hash in expected_eval_identities.items():
            result = split_results[split]
            if result["fixed_question_identity_sha256"] != expected_hash:
                raise ValueError(f"Fixed {split} question identity differs from Experiment 15")
        train_count = split_results["train"]["rows"]
        if train_count != 728_986:
            raise ValueError(f"Eligible training population changed: expected 728986, got {train_count}")
        cache_manifest = {
            **expected,
            "splits": split_results,
            "source_subtask_training_eligible_unique_tasks": source_manifest[
                "subtask_training_eligible"
            ],
            "sft_training_population": {
                "parent_rows": 500_000,
                "eligible_standalone_subtask_rows": train_count - 500_000,
                "total_rows": train_count,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(cache_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return destination, cache_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
