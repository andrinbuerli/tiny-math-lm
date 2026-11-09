"""Inspection and streaming access for the pinned English HF dataset split."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

DATASET_ID = "d0rj/mathematics_dataset"
DATASET_CONFIG = "default"
ENGLISH_SPLIT = "en"
EXPECTED_FIELDS = {"question", "answer", "category", "module"}
SELECTED_MODULES = (
    "arithmetic__add_or_sub",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__mixed",
    "algebra__linear_1d",
)
TRAIN_CATEGORIES = ("train-easy", "train-medium", "train-hard")
EVALUATION_CATEGORIES = ("interpolate", "extrapolate")
ALLOWED_CATEGORIES = frozenset((*TRAIN_CATEGORIES, *EVALUATION_CATEGORIES))


@dataclass(frozen=True)
class DatasetInspection:
    dataset_id: str
    config: str
    split: str
    revision: str
    features: dict[str, str]
    split_examples: int
    split_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_workspace_cache() -> None:
    cache_root = "/workspace/data/cache/huggingface"
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", f"{cache_root}/datasets")
    os.environ.setdefault("HF_HUB_CACHE", f"{cache_root}/hub")


def inspect_dataset(dataset_id: str = DATASET_ID, revision: str | None = None) -> DatasetInspection:
    """Resolve and validate the repository's live config/schema before reading rows."""
    _set_workspace_cache()
    from datasets import get_dataset_config_names, load_dataset_builder
    from huggingface_hub import HfApi

    immutable_revision = revision or HfApi().dataset_info(dataset_id).sha
    names = get_dataset_config_names(dataset_id, revision=immutable_revision)
    if DATASET_CONFIG not in names:
        raise ValueError(f"Dataset {dataset_id} has no expected {DATASET_CONFIG!r} config: {names}")
    builder = load_dataset_builder(dataset_id, name=DATASET_CONFIG, revision=immutable_revision)
    features = {name: str(feature) for name, feature in builder.info.features.items()}
    if set(features) != EXPECTED_FIELDS:
        raise ValueError(f"Unexpected dataset schema: {sorted(features)}")
    if ENGLISH_SPLIT not in builder.info.splits:
        raise ValueError(f"Dataset does not contain expected English split {ENGLISH_SPLIT!r}")
    split_info = builder.info.splits[ENGLISH_SPLIT]
    return DatasetInspection(
        dataset_id=dataset_id,
        config=DATASET_CONFIG,
        split=ENGLISH_SPLIT,
        revision=immutable_revision,
        features=features,
        split_examples=split_info.num_examples,
        split_bytes=split_info.num_bytes,
    )


def stream_english_rows(
    *, dataset_id: str = DATASET_ID, revision: str, modules: tuple[str, ...] = SELECTED_MODULES
) -> Iterator[dict[str, str]]:
    """Yield only selected-module English rows while retaining category labels."""
    _set_workspace_cache()
    from datasets import load_dataset

    source = load_dataset(
        dataset_id,
        name=DATASET_CONFIG,
        split=ENGLISH_SPLIT,
        revision=revision,
        streaming=True,
    )
    selected = set(modules)
    for row in source:
        module = row["module"]
        if module in selected:
            yield {field: str(row[field]) for field in EXPECTED_FIELDS}
