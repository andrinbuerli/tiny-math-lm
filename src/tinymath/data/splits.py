"""Deterministic module-balanced, leakage-free data pool construction."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from tinymath.data.load import (
    ALLOWED_CATEGORIES,
    SELECTED_MODULES,
    TRAIN_CATEGORIES,
)
from tinymath.data.prompts import format_prompt

_DIGIT_RUN = re.compile(r"\d+")
_OPERATOR = re.compile(r"\*\*|[+*/=\-]")
_CATEGORY_PRIORITY = {category: 0 for category in TRAIN_CATEGORIES}
_CATEGORY_PRIORITY.update({"interpolate": 2, "extrapolate": 3})


@dataclass(frozen=True)
class DataPools:
    splits: dict[str, list[dict[str, str]]]
    raw_counts: dict[str, dict[str, int]]
    duplicate_questions: int
    thresholds: dict[str, dict[str, int]]
    excluded_overlength: dict[str, dict[str, int]]


def normalize_question(question: str) -> str:
    """Normalize only Unicode canonical forms and whitespace for duplicate keys."""
    canonical = unicodedata.normalize("NFC", question)
    return " ".join(canonical.split())


def question_features(question: str) -> tuple[int, int]:
    """Return longest decimal operand width and math operator count."""
    digit_width = max((len(match.group()) for match in _DIGIT_RUN.finditer(question)), default=0)
    operator_count = len(_OPERATOR.findall(question))
    return digit_width, operator_count


def _stable_order(record: dict[str, str], seed: int) -> bytes:
    key = normalize_question(record["question"])
    return hashlib.sha256(f"{seed}\0{record['module']}\0{key}".encode()).digest()


def _quantile(values: list[int], quantile: float = 0.95) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(quantile * len(ordered)))
    return ordered[index]


def split_records(
    records: Iterable[dict[str, str]],
    *,
    sft_examples: int,
    validation_fraction: float = 0.05,
    rlvr_fraction: float = 0.05,
    seed: int = 17,
    context_length: int = 256,
    modules: tuple[str, ...] = SELECTED_MODULES,
) -> DataPools:
    """Dedupe across all source categories, then allocate disjoint fixed pools."""
    if sft_examples <= 0 or validation_fraction < 0 or rlvr_fraction < 0 or context_length < 1:
        raise ValueError("Pool sizes must be positive and split fractions nonnegative")
    if validation_fraction + rlvr_fraction >= 1:
        raise ValueError("Validation and RLVR fractions must sum to less than one")
    module_set = set(modules)
    raw_counts: Counter[tuple[str, str]] = Counter()
    unique: dict[str, dict[str, str]] = {}
    duplicate_questions = 0
    for record in records:
        module = record["module"]
        if module not in module_set:
            continue
        category = record["category"]
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Unexpected category {category!r} for selected module {module!r}")
        if not record["question"].strip() or not record["answer"].strip():
            continue
        raw_counts[(module, category)] += 1
        key = normalize_question(record["question"])
        previous = unique.get(key)
        if previous is None:
            unique[key] = record
            continue
        duplicate_questions += 1
        if _CATEGORY_PRIORITY[category] > _CATEGORY_PRIORITY[previous["category"]]:
            unique[key] = record

    excluded_overlength: Counter[tuple[str, str]] = Counter()
    eligible: dict[str, dict[str, str]] = {}
    for key, record in unique.items():
        sequence_length = len(format_prompt(record["question"])) + len(record["answer"]) + 2
        if sequence_length > context_length:
            excluded_overlength[(record["module"], record["category"])] += 1
        else:
            eligible[key] = record

    train_by_module = {module: [] for module in modules}
    for record in eligible.values():
        if record["category"] in TRAIN_CATEGORIES:
            train_by_module[record["module"]].append(record)

    n_modules = len(modules)
    train_counts = [sft_examples // n_modules + (i < sft_examples % n_modules) for i in range(n_modules)]
    validation_count = round(sft_examples * validation_fraction)
    rlvr_count = round(sft_examples * rlvr_fraction)
    validation_counts = [validation_count // n_modules + (i < validation_count % n_modules) for i in range(n_modules)]
    rlvr_counts = [rlvr_count // n_modules + (i < rlvr_count % n_modules) for i in range(n_modules)]

    splits: dict[str, list[dict[str, str]]] = {
        "sft_train": [],
        "sft_validation": [],
        "rlvr": [],
        "test": [],
        "extrapolate_other": [],
        "number_length_ood": [],
        "expression_length_ood": [],
    }
    for module_index, module in enumerate(modules):
        candidates = sorted(train_by_module[module], key=lambda row: _stable_order(row, seed))
        requested = train_counts[module_index] + validation_counts[module_index] + rlvr_counts[module_index]
        if len(candidates) < requested:
            raise ValueError(
                f"Module {module!r} has {len(candidates)} unique train examples, "
                f"but the configured pools require {requested}"
            )
        train_end = train_counts[module_index]
        validation_end = train_end + validation_counts[module_index]
        rlvr_end = validation_end + rlvr_counts[module_index]
        splits["sft_train"].extend(candidates[:train_end])
        splits["sft_validation"].extend(candidates[train_end:validation_end])
        splits["rlvr"].extend(candidates[validation_end:rlvr_end])

    splits["test"] = [
        record for record in eligible.values() if record["category"] == "interpolate"
    ]

    thresholds: dict[str, dict[str, int]] = {}
    for module in modules:
        reference = [row for row in splits["test"] if row["module"] == module]
        if not reference:
            raise ValueError(f"No official interpolate evaluation examples for {module!r}")
        digit_threshold = _quantile([question_features(row["question"])[0] for row in reference])
        expression_threshold = _quantile([question_features(row["question"])[1] for row in reference])
        thresholds[module] = {
            "number_digit_width_p95": digit_threshold,
            "expression_operator_count_p95": expression_threshold,
        }

    for record in eligible.values():
        if record["category"] != "extrapolate":
            continue
        digit_width, operator_count = question_features(record["question"])
        threshold = thresholds[record["module"]]
        longer_number = digit_width > threshold["number_digit_width_p95"]
        longer_expression = operator_count > threshold["expression_operator_count_p95"]
        if longer_number and not longer_expression:
            splits["number_length_ood"].append(record)
        elif longer_expression and not longer_number:
            splits["expression_length_ood"].append(record)
        else:
            splits["extrapolate_other"].append(record)

    for values in splits.values():
        values.sort(key=lambda row: (row["module"], _stable_order(row, seed)))
    return DataPools(
        splits=splits,
        raw_counts={
            module: {category: raw_counts[(module, category)] for category in sorted(ALLOWED_CATEGORIES)}
            for module in modules
        },
        duplicate_questions=duplicate_questions,
        thresholds=thresholds,
        excluded_overlength={
            module: {
                category: excluded_overlength[(module, category)]
                for category in sorted(ALLOWED_CATEGORIES)
            }
            for module in modules
        },
    )
