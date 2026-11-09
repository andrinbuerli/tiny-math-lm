from __future__ import annotations

import json

from tiny_math_lm.data.load import SELECTED_MODULES, DatasetInspection
from tiny_math_lm.data.prepare import prepare_bundle
from tiny_math_lm.data.splits import normalize_question, split_records


def _records():
    rows = []
    for module in SELECTED_MODULES:
        for index in range(1, 6):
            rows.append(
                {
                    "question": f"{module} unique {index} + 1",
                    "answer": str(index + 1),
                    "category": "interpolate",
                    "module": module,
                }
            )
        for index, category in enumerate(
            ("train-easy", "train-medium", "train-hard", "train-easy", "train-hard")
        ):
            rows.append(
                {
                    "question": (
                        f"{module} unique {category} {index} 1234567890 + 0"
                        if index == 4
                        else f"{module} unique {category} {index} 8 + 1"
                    ),
                    "answer": "1234567890" if index == 4 else "9",
                    "category": category,
                    "module": module,
                }
            )
        rows.extend(
            [
                {
                    "question": f"{module} unique 123456 + 1",
                    "answer": "123457",
                    "category": "extrapolate",
                    "module": module,
                },
                {
                    "question": f"{module} unique 1 + 1 + 1 + 1 + 1",
                    "answer": "5",
                    "category": "extrapolate",
                    "module": module,
                },
                {
                    "question": f"{module} unique 123456 + 1 + 1 + 1 + 1",
                    "answer": "123460",
                    "category": "extrapolate",
                    "module": module,
                },
            ]
        )
    # The official evaluation copy wins, so this item cannot enter training.
    rows.append(
        {
            "question": f"{SELECTED_MODULES[0]} unique 1   + 1",
            "answer": "2",
            "category": "train-easy",
            "module": SELECTED_MODULES[0],
        }
    )
    rows.append(
        {
            "question": "ignored module problem",
            "answer": "0",
            "category": "train-easy",
            "module": "geometry__triangle",
        }
    )
    return rows


def test_splits_are_module_balanced_disjoint_and_eval_wins_dedup():
    pools = split_records(
        _records(), sft_examples=5, validation_fraction=0.2, rlvr_fraction=0.2, seed=42
    )
    assert len(pools.splits["sft_train"]) == 5
    assert len(pools.splits["sft_validation"]) == 1
    assert len(pools.splits["rlvr"]) == 1
    assert all(
        sum(row["module"] == module for row in pools.splits["sft_train"]) == 1
        for module in SELECTED_MODULES
    )
    assert pools.duplicate_questions == 1
    seen = set()
    for split in pools.splits.values():
        keys = {normalize_question(row["question"]) for row in split}
        assert not seen.intersection(keys)
        seen.update(keys)
    assert normalize_question("  1   +\n 2 ") == "1 + 2"


def test_controlled_ood_splits_vary_one_question_feature():
    pools = split_records(
        _records(), sft_examples=5, validation_fraction=0.2, rlvr_fraction=0.2
    )
    assert len(pools.splits["number_length_ood"]) == len(SELECTED_MODULES)
    assert len(pools.splits["expression_length_ood"]) == len(SELECTED_MODULES)
    assert len(pools.splits["extrapolate_other"]) == len(SELECTED_MODULES)
    for module in SELECTED_MODULES:
        threshold = pools.thresholds[module]
        number_case = next(row for row in pools.splits["number_length_ood"] if row["module"] == module)
        expr_case = next(
            row for row in pools.splits["expression_length_ood"] if row["module"] == module
        )
        assert len("123456") > threshold["number_digit_width_p95"]
        assert len("1") <= threshold["number_digit_width_p95"]
        assert number_case["category"] == expr_case["category"] == "extrapolate"


def test_missing_official_extrapolate_uses_reserved_disjoint_train_rows():
    rows = [row for row in _records() if row["category"] != "extrapolate"]
    for module in SELECTED_MODULES:
        rows.extend(
            [
                {
                    "question": f"{module} held out number 123456789012 + 1",
                    "answer": "123456789013",
                    "category": "train-medium",
                    "module": module,
                },
                {
                    "question": f"{module} held out expression 1 + 1 + 1 + 1 + 1",
                    "answer": "5",
                    "category": "train-hard",
                    "module": module,
                },
            ]
        )
    pools = split_records(
        rows,
        sft_examples=5,
        validation_fraction=0.2,
        rlvr_fraction=0.2,
        ood_examples_per_module=1,
    )
    assert len(pools.splits["number_length_ood"]) == len(SELECTED_MODULES)
    assert len(pools.splits["expression_length_ood"]) == len(SELECTED_MODULES)
    training_keys = {
        normalize_question(row["question"])
        for name in ("sft_train", "sft_validation", "rlvr")
        for row in pools.splits[name]
    }
    for module in SELECTED_MODULES:
        assert "train categories" in pools.ood_source_by_module[module]
    for name in ("number_length_ood", "expression_length_ood"):
        assert all(row["category"].startswith("train-") for row in pools.splits[name])
        assert not training_keys.intersection(
            normalize_question(row["question"]) for row in pools.splits[name]
        )


def test_prepared_bundle_persists_tokenizer_samples_and_hashed_manifest(workspace_tmpdir):
    inspection = DatasetInspection(
        dataset_id="d0rj/mathematics_dataset",
        config="default",
        split="en",
        revision="abc123",
        features={"question": "string", "answer": "string", "category": "string", "module": "string"},
        split_examples=100,
        split_bytes=1000,
    )
    pools, tokenizer, manifest = prepare_bundle(
        _records(),
        inspection,
        output_dir=workspace_tmpdir / "data",
        seed=19,
        sft_examples=5,
        validation_fraction=0.2,
        rlvr_fraction=0.2,
        context_length=256,
    )
    assert manifest["tokenizer"]["sha256"] == tokenizer.sha256
    assert manifest["splits"]["sft_train"]["examples"] == len(pools.splits["sft_train"])
    assert len(manifest["split_manifest_sha256"]) == 64
    assert (workspace_tmpdir / "data" / "char_tokenizer.json").exists()
    samples = json.loads((workspace_tmpdir / "data" / "inspected_samples.json").read_text())
    assert samples["sft_train"]
