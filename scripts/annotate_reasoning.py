"""Hydra entrypoint that writes a verified reasoning-reference data bundle."""

from __future__ import annotations

import hashlib
import json
import logging
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from tinymath.config import workspace_path
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.data.reasoning import (
    DELEGATION_POLICY,
    SUBTASK_SCHEMA_VERSION,
    annotate_row,
    distribution,
    verify_annotation,
)
from tinymath.data.reasoning_v3_bundle import build_reasoning_v3_bundle
from tinymath.tracking.wandb import finish_wandb, initialize_wandb

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _load_split(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _assert_known_characters(tokenizer: CharacterTokenizer, text: str, context: str) -> None:
    """Fail explicitly if a split contains characters absent from the frozen vocab."""
    ids = tokenizer.encode(text, allow_unk=True)
    if tokenizer.unk_id in ids:
        unknown = next(character for character, token_id in zip(text, ids) if token_id == tokenizer.unk_id)
        raise ValueError(f"Unknown tokenizer character in {context}: U+{ord(unknown):04X} {unknown!r}")


@hydra.main(version_base="1.3", config_path="../configs/annotation", config_name="default")
def main(config: DictConfig) -> None:
    source = Path(str(config.source_bundle)).resolve()
    destination = Path(str(config.output_bundle)).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source bundle does not exist: {source}")
    if source == destination or source in destination.parents:
        raise ValueError("The derived bundle must not overwrite or nest inside its source bundle")
    manifest_path = source / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_info = source_manifest.get("splits")
    if not isinstance(split_info, dict) or not split_info:
        raise ValueError("Source manifest has no split inventory")
    run_dir = workspace_path(
        "runs", str(config.name), "data-annotation", f"seed{int(config.seed)}"
    )
    run = initialize_wandb(config, run_dir, job_type="data-annotation")
    try:
        if str(config.protocol) == "reasoning_v3":
            if int(config.get("subtask_schema_version", SUBTASK_SCHEMA_VERSION)) != SUBTASK_SCHEMA_VERSION:
                raise ValueError("Unsupported v3 subtask schema version")
            if config.get("delegation_policy", DELEGATION_POLICY) != DELEGATION_POLICY:
                raise ValueError("Unsupported v3 delegation policy")
            result = build_reasoning_v3_bundle(
                source, destination, max_tokens=int(config.max_tokens),
                workers=int(config.get("workers", 8)),
                progress=lambda stage, split, rows: run.log({f"annotation/progress/{stage}/{split}": rows}),
            )
            run.config.update({
                "source_manifest_sha256": result["source_manifest_sha256"],
                "annotation_sha256": result["annotation"]["annotation_sha256"],
                "annotation_output_bundle": str(destination),
                "subtask_schema_version": result["subtask_schema_version"],
                "subtask_operand_semantics": result["subtask_operand_semantics"],
                "delegation_policy": result["delegation_policy"],
                "split_counts": {
                    name: value["examples"] for name, value in result["splits"].items()
                },
                "parent_split_counts": {
                    name: value["parent_examples"] for name, value in result["splits"].items()
                },
                "subtask_split_counts": {
                    name: value["subtask_examples"] for name, value in result["splits"].items()
                },
            }, allow_val_change=True)
            run.log({
                "annotation/parent_rows": sum(v["parent_examples"] for v in result["splits"].values()),
                "annotation/subtask_rows": sum(v["subtask_examples"] for v in result["splits"].values()),
                "annotation/eligible_subtask_tasks": result["subtask_training_eligible_tasks"],
            })
            LOGGER.info("Wrote reasoning_v3 bundle to %s", destination)
            return
        # Fit on SFT training questions, answers and complete reference traces,
        # plus the fixed trace grammar. Held-out scratchpads are only checked
        # against this frozen vocabulary in the second pass.
        train_name = "sft_train"
        trace_grammar = (
            "0123456789+-*/(),.:;=@^> " + string.ascii_lowercase + string.ascii_uppercase
        )
        tokenizer_characters = set(trace_grammar)
        source_hashes: dict[str, str] = {}
        for split_name, metadata in split_info.items():
            source_path = source / str(metadata["path"])
            source_hashes[split_name] = _sha256(source_path)
            for row in _load_split(source_path):
                if split_name == train_name:
                    tokenizer_characters.update(row["question"])
                    tokenizer_characters.update(row["answer"])
                full = annotate_row(
                    row, max_tokens=10**9, protocol=str(config.protocol)
                )
                if split_name == train_name:
                    tokenizer_characters.update(str(full["scratchpad"]))
        tokenizer = CharacterTokenizer(tokenizer_characters, protocol=str(config.protocol))

        def exact_length(row: dict[str, str], scratchpad: str) -> int:
            prompt = tokenizer.encode(
                format_prompt(row["question"], protocol=str(config.protocol)), add_bos=True
            )
            completion = tokenizer.encode_completion(row["answer"], scratchpad=scratchpad)
            return len(prompt) + len(completion)

        # The destination is a dedicated derived bundle. Rewriting it from the
        # immutable source makes interrupted annotation rerunnable under the
        # same W&B identity without retaining a misleading partial manifest.
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer_path = destination / "char_tokenizer.json"
        tokenizer.save(tokenizer_path)
        split_manifest: dict[str, dict[str, Any]] = {}
        group_lengths: dict[tuple[str, str], list[int]] = defaultdict(list)
        group_reduced: Counter[tuple[str, str]] = Counter()
        collapsed: Counter[str] = Counter()
        for split_name, metadata in split_info.items():
            source_path = source / str(metadata["path"])
            source_hash = source_hashes[split_name]
            output_path = destination / f"{split_name}.jsonl"
            annotated_digest = hashlib.sha256()
            row_count = 0
            with output_path.open("w", encoding="utf-8") as handle:
                for index, row in enumerate(_load_split(source_path)):
                    try:
                        annotated = annotate_row(
                            row,
                            max_tokens=int(config.max_tokens),
                            length_fn=exact_length,
                            protocol=str(config.protocol),
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"Annotation failed at {split_name}[{index}] "
                            f"({row.get('module')}): {row.get('question')!r}: {exc}"
                        ) from exc
                    if any(annotated.get(key) != row.get(key) for key in row):
                        raise AssertionError(f"Source fields changed at {split_name}[{index}]")
                    try:
                        verify_annotation(row, annotated)
                    except Exception as exc:
                        raise ValueError(
                            f"Trace replay failed at {split_name}[{index}] "
                            f"({row.get('module')}): {row.get('question')!r}: {exc}"
                        ) from exc
                    encoded_line = _json_line(annotated)
                    handle.write(encoded_line)
                    annotated_digest.update(encoded_line.encode("utf-8"))
                    row_count += 1
                    group = (str(row["module"]), split_name)
                    group_lengths[group].append(int(annotated["retained_length"]))
                    if bool(annotated["reduced"]):
                        group_reduced[group] += 1
                    collapsed.update(annotated["collapsed_categories"])
                    _assert_known_characters(tokenizer, row["question"], f"{split_name}[{index}].question")
                    _assert_known_characters(tokenizer, row["answer"], f"{split_name}[{index}].answer")
                    _assert_known_characters(
                        tokenizer, str(annotated["scratchpad"]), f"{split_name}[{index}].scratchpad"
                    )
            split_manifest[split_name] = {
                "path": output_path.name,
                "examples": row_count,
                "source_sha256": source_hash,
                "sha256": _sha256(output_path),
                "annotation_sha256": annotated_digest.hexdigest(),
            }
        tokenizer_hash = tokenizer.sha256
        stats = {}
        for (module, split), values in sorted(group_lengths.items()):
            stats[f"{module}/{split}"] = {
                **distribution(values),
                "reduced_count": group_reduced[(module, split)],
                "reduced_fraction": group_reduced[(module, split)] / len(values),
            }
        training_pool_splits = {"sft_train", "sft_validation", "rlvr"}
        training_pool_counts: Counter[str] = Counter()
        training_pool_lengths: list[int] = []
        for (module, split), values in group_lengths.items():
            if split in training_pool_splits:
                training_pool_counts[module] += sum(values)
                training_pool_lengths.extend(values)
        if len(training_pool_lengths) != 550000:
            raise ValueError(
                "Expected 550,000 SFT/validation/RLVR tokenized rows, "
                f"found {len(training_pool_lengths)}"
            )
        annotation_payload = {
            "protocol": str(config.protocol),
            "trace_version": str(config.protocol),
            "max_tokens": int(config.max_tokens),
            "source_bundle": str(source),
            "source_manifest_sha256": _sha256(manifest_path),
            "source_tokenizer_sha256": source_manifest.get("tokenizer", {}).get("sha256"),
            "source_split_sha256": source_hashes,
            "tokenizer_character_set_sha256": tokenizer_hash,
            "tokenizer_sha256": tokenizer_hash,
            "tokenizer_path": tokenizer_path.name,
            "splits": split_manifest,
            "module_split_statistics": stats,
        }
        annotation_payload["annotation_sha256"] = hashlib.sha256(
            json.dumps(annotation_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        derived = dict(source_manifest)
        derived["experiment_identity"] = str(config.name)
        derived["source_bundle"] = str(source)
        derived["source_manifest_sha256"] = _sha256(manifest_path)
        derived["context_length"] = int(config.max_tokens)
        derived["prompt_format"] = "Q: {original_question}\\n"
        derived["completion_format"] = "<scp>{scratchpad}</scp>\\nA: {original_answer}<EOS>"
        derived["annotation"] = annotation_payload
        derived["splits"] = split_manifest
        derived["token_counts_by_module_training_pools"] = dict(sorted(training_pool_counts.items()))
        derived["token_length_distribution_training_pools"] = distribution(training_pool_lengths)
        derived["tokenizer"] = {
            "version": (
                5 if str(config.protocol) == "reasoning_v4"
                else 3 if str(config.protocol) == "reasoning_v2"
                else 2
            ),
            "protocol": str(config.protocol),
            "strict_single_unicode_codepoint": True,
            "special_tokens": list(tokenizer.special_tokens),
            "vocab_size": tokenizer.vocab_size,
            "sha256": tokenizer.sha256,
            "path": tokenizer_path.name,
        }
        (destination / "manifest.json").write_text(
            json.dumps(derived, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote %d annotated rows to %s", sum(v["examples"] for v in split_manifest.values()), destination)
        run.config.update({
            "source_manifest_sha256": annotation_payload["source_manifest_sha256"],
            "annotation_sha256": annotation_payload["annotation_sha256"],
            "tokenizer_character_set_sha256": tokenizer_hash,
            "annotation_output_bundle": str(destination),
            "split_counts": {name: value["examples"] for name, value in split_manifest.items()},
        }, allow_val_change=True)
        run.log({"annotation/rows": sum(v["examples"] for v in split_manifest.values()),
                 "annotation/collapsed_operations": sum(collapsed.values()),
                 "annotation/length_stats": stats,
                 **{f"annotation/collapsed/{name}": count for name, count in collapsed.items()}})
    finally:
        finish_wandb(run, exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
    main()
