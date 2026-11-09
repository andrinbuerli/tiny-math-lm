"""Create an auditable derived bundle excluding specifically verified bad rows."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from tiny_math_lm.config import workspace_path
from tiny_math_lm.data.reasoning import verify_annotation

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def filter_bundle(
    *,
    source: Path,
    destination: Path,
    expected_manifest_sha256: str,
    expected_error: str,
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Copy a bundle while dropping only rows whose identity and verifier error match."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("The destination must be separate from the source bundle")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a derived bundle: {destination}")

    manifest_path = source / "manifest.json"
    source_manifest_digest = _sha256(manifest_path)
    if source_manifest_digest != expected_manifest_sha256:
        raise ValueError(
            f"Source manifest hash mismatch: {source_manifest_digest} != {expected_manifest_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_split: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for item in exclusions:
        split, line_number = str(item["split"]), int(item["line"])
        if line_number < 1 or line_number in by_split[split]:
            raise ValueError(f"Invalid or duplicate exclusion line {split}:{line_number}")
        by_split[split][line_number] = dict(item)

    shutil.copytree(source, destination)
    try:
        updated_splits: dict[str, dict[str, Any]] = {}
        audit: list[dict[str, Any]] = []
        for split, spec in manifest["splits"].items():
            source_file = source / str(spec["path"])
            destination_file = destination / str(spec["path"])
            if _sha256(source_file) != spec["sha256"]:
                raise ValueError(f"Source split hash mismatch for {split}")
            excluded = by_split.get(split, {})
            counts: Counter[str] = Counter()
            digest = hashlib.sha256()
            with source_file.open("rb") as src, destination_file.open("wb") as dst:
                for line_number, raw in enumerate(src, 1):
                    if not raw.strip():
                        dst.write(raw)
                        continue
                    row = json.loads(raw)
                    item = excluded.get(line_number)
                    if item is not None:
                        expected = (str(item["module"]), str(item["question"]), str(item["answer"]))
                        actual = (str(row.get("module", "")), str(row.get("question", "")), str(row.get("answer", "")))
                        if actual != expected or not row.get("is_sub_task") or row.get("trace_version") != "reasoning_v2":
                            raise ValueError(f"Exclusion identity mismatch at {split}:{line_number}")
                        try:
                            verify_annotation(row, row)
                        except ValueError as exc:
                            if str(exc) != expected_error:
                                raise ValueError(f"Unexpected verifier error at {split}:{line_number}: {exc}") from exc
                        else:
                            raise ValueError(f"Exclusion unexpectedly verifies at {split}:{line_number}")
                        audit.append({
                            "split": split,
                            "line": line_number,
                            "module": actual[0],
                            "question": actual[1],
                            "answer": actual[2],
                            "trace_version": row["trace_version"],
                            "source_row_sha256": _row_digest(raw),
                            "verifier_error": expected_error,
                        })
                        continue
                    dst.write(raw)
                    counts["rows"] += 1
                    counts["subtasks" if row.get("is_sub_task") else "parents"] += 1
                    digest.update(raw)
            if set(excluded) != {entry["line"] for entry in audit if entry["split"] == split}:
                raise ValueError(f"Not all exclusions were applied for {split}")
            updated = dict(spec)
            updated.update({
                "examples": counts["rows"],
                "parent_examples": counts["parents"],
                "subtask_examples": counts["subtasks"],
                "sha256": digest.hexdigest(),
                "annotation_sha256": digest.hexdigest(),
            })
            updated_splits[split] = updated

        unknown_splits = set(by_split) - set(manifest["splits"])
        if unknown_splits or len(audit) != len(exclusions):
            raise ValueError(f"Unapplied exclusions: splits={sorted(unknown_splits)}, rows={len(audit)}/{len(exclusions)}")
        manifest["source_split_manifest_sha256"] = manifest["split_manifest_sha256"]
        manifest["splits"] = updated_splits
        manifest["split_manifest_sha256"] = hashlib.sha256(
            json.dumps(updated_splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest["derived_filter"] = {
            "operation": "exclude_failed_reasoning_v2_rows",
            "source_bundle": str(source),
            "source_manifest_sha256": source_manifest_digest,
            "excluded_row_count": len(audit),
            "excluded_rows": audit,
        }
        annotation = dict(manifest.get("annotation", {}))
        annotation["splits"] = updated_splits
        manifest["annotation"] = annotation
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        LOGGER.info(
            "Created %s from %s; excluded %d rows; manifest SHA256 %s",
            destination,
            source,
            len(audit),
            _sha256(destination / "manifest.json"),
        )
        return manifest
    except BaseException:
        shutil.rmtree(destination)
        raise


@hydra.main(version_base="1.3", config_path="../configs", config_name="filter_invalid_reasoning_bundle")
def main(config: DictConfig) -> None:
    filter_bundle(
        source=workspace_path(str(config.source_bundle)),
        destination=workspace_path(str(config.output_bundle)),
        expected_manifest_sha256=str(config.expected_manifest_sha256),
        expected_error=str(config.expected_error),
        exclusions=OmegaConf.to_container(config.exclusions, resolve=True),
    )


if __name__ == "__main__":
    main()
