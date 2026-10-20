"""Fixed-population evaluation for the native-tokenizer Experiment 16 backend."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from tinymath.data.hf_cached_dataset import HFTokenSplit
from tinymath.data.hf_reasoning import identity_hash, score_native_reasoning, sha256_file
from tinymath.data.prompts import format_prompt
from tinymath.data.splits import question_features
from tinymath.model.hf_qwen import load_model, load_tokenizer, model_provenance
from tinymath.rewards.verify import verify_answer
from tinymath.train.hf_generation import _mask_non_token_logits, generate_native_rollout

FIXED_SPLIT_COUNTS = {"test": 50_000, "number_length_ood": 5_000, "expression_length_ood": 2_000}


def _cache_dir(config: Any) -> Path:
    root = Path("/workspace/data/cache/tokenized")
    revision = str(config.model.revision)
    candidates = sorted(root.glob(f"exp16_qwen3_base_reasoning_v3-{revision[:12]}/*/manifest.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one token cache for model revision, found {len(candidates)}")
    return candidates[0].parent


@torch.no_grad()
def _generate_qa(model: Any, tokenizer: Any, question: str, *, context: int, cap: int) -> tuple[str, int, bool]:
    prompt_ids = tokenizer.encode(format_prompt(question), add_special_tokens=False)
    room = min(cap, context - len(prompt_ids))
    if room < 1:
        return "", 0, True
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
    past = output.past_key_values
    logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
    generated: list[int] = []
    generated_token_count = 0
    stopped = False
    for offset in range(room):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id == tokenizer.eos_token_id:
            stopped = True
            break
        generated.append(token_id)
        generated_token_count += 1
        piece = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if "\n" in piece or "\r" in piece:
            generated = tokenizer.encode(piece.splitlines()[0], add_special_tokens=False)
            stopped = True
            break
        if offset + 1 == room:
            break
        new_length = len(prompt_ids) + offset + 1
        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        output = model(
            input_ids=token,
            attention_mask=torch.ones((1, new_length), dtype=torch.long, device=device),
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
    text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    answer = text.splitlines()[0] if text else ""
    return answer, generated_token_count, not stopped


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validate_checkpoint_identity(checkpoint: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    identity_path = checkpoint / "experiment_identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(f"Checkpoint provenance is missing: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    for key, actual in (
        ("model_revision", provenance["model_revision"]),
        ("model_config_sha256", provenance["model_config_sha256"]),
        ("tokenizer_sha256", provenance["tokenizer_sha256"]),
        ("parameters", provenance["model_parameters"]),
    ):
        if identity.get(key) != actual:
            raise ValueError(f"Checkpoint {key} does not match the pinned evaluation model")
    local_config = checkpoint / "config.json"
    if (
        not local_config.is_file()
        or sha256_file(local_config) != identity.get("checkpoint_config_sha256")
    ):
        raise ValueError("Checkpoint serialized config hash does not match its identity")
    weights = identity.get("model_weight_files", {})
    if not weights:
        raise ValueError("Checkpoint identity has no weight-file hashes")
    digest = hashlib.sha256()
    for filename, expected_hash in sorted(weights.items()):
        path = checkpoint / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Checkpoint weight file hash mismatch: {filename}")
        digest.update(filename.encode("utf-8"))
        digest.update(expected_hash.encode("ascii"))
    if digest.hexdigest() != identity.get("checkpoint_sha256"):
        raise ValueError("Checkpoint aggregate weight hash does not match its identity")
    return identity


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    operators: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["module"]].append(row)
        digits[str(row["digit_width"])].append(row)
        operators[str(row["operator_count"])].append(row)

    def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {
            "examples": n,
            "accuracy": _ratio(sum(bool(r["correct"]) for r in rows), n),
            "format_validity": _ratio(sum(bool(r["format_valid"]) for r in rows), n),
            "capped_completions": sum(bool(r["capped"]) for r in rows),
            "mean_output_tokens": _ratio(sum(int(r["output_tokens"]) for r in rows), n),
            "executor_calls": sum(int(r.get("call_count", 0)) for r in rows),
            "executor_failures": sum(bool(r.get("executor_failure")) for r in rows),
            "policy_tokens": sum(int(r.get("policy_tokens", r["output_tokens"])) for r in rows),
            "executor_tokens": sum(int(r.get("executor_tokens", 0)) for r in rows),
            "full_protocol_success": _ratio(sum(bool(r.get("full_success", False)) for r in rows), n),
            "trace_validity": _ratio(sum(bool(r.get("trace_valid", False)) for r in rows), n),
        }

    return {
        **score_rows(records),
        "by_module": {key: score_rows(value) for key, value in sorted(grouped.items())},
        "by_digit_width": {key: score_rows(value) for key, value in sorted(digits.items(), key=lambda p: int(p[0]))},
        "by_operator_count": {key: score_rows(value) for key, value in sorted(operators.items(), key=lambda p: int(p[0]))},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_index", "module", "question", "target", "prediction", "correct", "format_valid",
        "trace_valid", "full_success", "capped", "output_tokens", "digit_width", "operator_count",
        "call_count", "executor_failure", "policy_tokens", "executor_tokens", "termination_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def evaluate_hf_checkpoint(
    *, config: Any, bundle_dir: Path, output_dir: Path, checkpoint: Path | None,
    split_names: list[str], max_examples_per_split: int | None = None,
    run: Any = None,
) -> dict[str, Any]:
    """Evaluate ordinary Q/A and reasoning/executor formats on the same fixed rows."""
    cache_dir = _cache_dir(config)
    cache_manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_bundle = str(config.checkpoint_provenance.data_bundle_manifest_sha256)
    if sha256_file(bundle_dir / "manifest.json") != expected_bundle:
        raise ValueError("Evaluation source bundle manifest does not match Experiment 15 provenance")
    if cache_manifest["source_bundle_manifest_sha256"] != expected_bundle:
        raise ValueError("Evaluation token cache belongs to a different source bundle")
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model, checkpoint=checkpoint, training=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    provenance = model_provenance(config.model, tokenizer, model)
    checkpoint_identity = (
        _validate_checkpoint_identity(checkpoint, provenance) if checkpoint is not None else {}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    for split_name in split_names:
        split = HFTokenSplit(cache_dir, split_name)
        expected = cache_manifest["splits"][split_name]
        expected_count = FIXED_SPLIT_COUNTS[split_name]
        declared_identities = config.checkpoint_provenance.fixed_evaluation_question_identity_sha256
        if expected["fixed_question_identity_sha256"] != declared_identities[split_name]:
            raise ValueError(f"Fixed {split_name} population differs from Experiment 15 provenance")
        if (
            int(expected["fixed_question_identity_count"]) != expected_count
            or len(split) != expected_count
        ):
            raise ValueError(f"Fixed {split_name} population count changed")
        count = len(split) if max_examples_per_split is None else min(len(split), max_examples_per_split)
        all_rows = list(split.rows())
        digest, identity_count = identity_hash(all_rows)
        if identity_count != int(expected["fixed_question_identity_count"]) or digest != expected["fixed_question_identity_sha256"]:
            raise ValueError(f"Fixed {split_name} question identities changed")
        rows = all_rows[:count]
        for mode in ("ordinary_qa", "reasoning_executor"):
            records: list[dict[str, Any]] = []
            for index, row in enumerate(rows):
                digit_width, operator_count = question_features(str(row["question"]))
                base = {
                    "row_index": index,
                    "module": row["module"],
                    "question": row["question"],
                    "target": row["answer"],
                    "digit_width": digit_width,
                    "operator_count": operator_count,
                    "policy_tokens": 0,
                    "executor_tokens": 0,
                    "call_count": 0,
                    "executor_failure": False,
                    "trace_valid": False,
                    "full_success": False,
                }
                if mode == "ordinary_qa":
                    prediction, token_count, capped = _generate_qa(
                        model, tokenizer, str(row["question"]),
                        context=int(config.model.context_length), cap=int(config.eval.max_new_tokens),
                    )
                    correct, valid = verify_answer(prediction, str(row["answer"]), question=str(row["question"]))
                    records.append(base | {
                        "prediction": prediction, "correct": bool(correct and valid), "format_valid": valid,
                        "capped": capped, "output_tokens": token_count, "policy_tokens": token_count,
                        "termination_reason": "budget_exceeded" if capped else "newline_or_eos",
                    })
                else:
                    prompt_ids = tokenizer.encode(f"Q: {row['question']}\n", add_special_tokens=False)
                    rollout = generate_native_rollout(
                        model, tokenizer, prompt_ids, row,
                        context_length=int(config.model.context_length),
                        max_completion_tokens=int(config.eval.max_new_tokens),
                        temperature=0.0, rollout=config.rollout, greedy=True,
                    )
                    completion_ids = rollout.token_ids
                    if completion_ids and completion_ids[-1] == tokenizer.eos_token_id:
                        completion_ids = completion_ids[:-1]
                    text = tokenizer.decode(completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    scored = score_native_reasoning(
                        text, row,
                        capped=rollout.termination_reason == "budget_exceeded",
                        failure=rollout.termination_reason if rollout.termination_reason not in {"eos", "budget_exceeded"} else None,
                        format_weight=float(config.rlvr.reward_format_weight),
                        success_weight=float(config.rlvr.reward_success_weight),
                        max_calls=int(config.rollout.max_calls),
                        max_scalar_chars=int(config.rollout.max_scalar_chars),
                    )
                    records.append(base | {
                        "prediction": text, "correct": scored.answer_correct, "format_valid": scored.format_valid,
                        "trace_valid": scored.trace_valid, "full_success": scored.full_success,
                        "capped": rollout.termination_reason == "budget_exceeded",
                        "output_tokens": len(rollout.token_ids), "call_count": scored.call_count,
                        "executor_failure": rollout.termination_reason not in {"eos", "budget_exceeded"},
                        "policy_tokens": rollout.policy_tokens, "executor_tokens": rollout.executor_tokens,
                        "termination_reason": rollout.termination_reason,
                    })
            summary = _summarize(records)
            key = f"{split_name}/{mode}"
            all_results[key] = summary
            _write_csv(output_dir / f"{split_name}_{mode}.csv", records)
            if run is not None:
                run.log({f"eval/{key}/{metric}": value for metric, value in summary.items() if isinstance(value, (int, float)) and value is not None})
                for module, metrics in summary["by_module"].items():
                    run.log({f"eval/{key}/module/{module}/{metric}": value for metric, value in metrics.items() if isinstance(value, (int, float)) and value is not None})
    result = {
        "experiment": str(config.experiment.name),
        "checkpoint": str(checkpoint) if checkpoint else "pinned_pristine_base",
        "checkpoint_sha256": checkpoint_identity.get("checkpoint_sha256"),
        "checkpoint_identity": checkpoint_identity,
        "model": provenance,
        "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "source_bundle_manifest_sha256": cache_manifest["source_bundle_manifest_sha256"],
        "splits": all_results,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
