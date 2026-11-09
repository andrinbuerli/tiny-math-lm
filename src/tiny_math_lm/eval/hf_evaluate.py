"""Fixed-population evaluation for the native-tokenizer Experiment 16 backend."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from tiny_math_lm.data.hf_cached_dataset import HFTokenSplit
from tiny_math_lm.data.hf_reasoning import identity_hash, score_native_reasoning, sha256_file
from tiny_math_lm.data.prompts import format_prompt
from tiny_math_lm.data.splits import question_features
from tiny_math_lm.model.hf_qwen import load_model, load_tokenizer, model_provenance
from tiny_math_lm.rewards.verify import verify_answer
from tiny_math_lm.train.hf_generation import (
    _mask_non_token_logits,
    generate_native_rollout_batch,
)

FIXED_SPLIT_COUNTS = {"test": 50_000, "number_length_ood": 5_000, "expression_length_ood": 2_000}
_LINE_TERMINATOR_IDS: dict[int, tuple[Any, frozenset[int]]] = {}


def _cache_dir(config: Any) -> Path:
    root = Path("/workspace/data/cache/tokenized")
    revision = str(config.model.revision)
    candidates = sorted(root.glob(f"exp16_qwen3_base_reasoning_v3-{revision[:12]}/*/manifest.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one token cache for model revision, found {len(candidates)}")
    return candidates[0].parent


def _select_cache_rows(cache: Any, indices: torch.Tensor) -> Any:
    select = getattr(cache, "batch_select_indices", None)
    if callable(select):
        select(indices)
        return cache
    if isinstance(cache, tuple):
        return tuple(
            tuple(value.index_select(0, indices) if isinstance(value, torch.Tensor) else value for value in layer)
            for layer in cache
        )
    raise TypeError("Batched Q/A generation requires a cache that supports batch row selection")


def _line_terminator_token_ids(tokenizer: Any) -> frozenset[int]:
    cached = _LINE_TERMINATOR_IDS.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    token_ids = set()
    for token_id in range(len(tokenizer)):
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if "\n" in decoded or "\r" in decoded:
            token_ids.add(token_id)
    token_ids = frozenset(token_ids)
    _LINE_TERMINATOR_IDS[id(tokenizer)] = (tokenizer, token_ids)
    return token_ids


@torch.no_grad()
def _generate_qa_batch(
    model: Any, tokenizer: Any, questions: list[str], *, context: int, cap: int,
) -> list[tuple[str, int, bool]]:
    if not questions:
        return []
    prompt_ids = [tokenizer.encode(format_prompt(question), add_special_tokens=False) for question in questions]
    rooms = [min(cap, context - len(ids)) for ids in prompt_ids]
    results: list[tuple[str, int, bool] | None] = [None] * len(questions)
    for index, room in enumerate(rooms):
        if room < 1:
            results[index] = ("", 0, True)
    active_indices = [index for index, room in enumerate(rooms) if room > 0]
    if not active_indices:
        return [result for result in results if result is not None]
    active_count = len(active_indices)
    prompt_ids = [prompt_ids[index] for index in active_indices]
    rooms = [rooms[index] for index in active_indices]
    active_batch_indices = list(active_indices)
    device = next(model.parameters()).device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = tokenizer.eos_token_id if pad_id is None else pad_id
    line_terminator_ids = _line_terminator_token_ids(tokenizer)
    width = max(len(ids) for ids in prompt_ids)
    input_ids = torch.full((active_count, width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(prompt_ids):
        input_ids[index, width - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, width - len(ids):] = 1
    position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past = output.past_key_values
    logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
    generated: list[list[int]] = [[] for _ in questions]
    counts = [0] * len(questions)
    stopped = [True] * len(questions)
    for index in active_indices:
        stopped[index] = False
    for offset in range(max(rooms, default=0)):
        token_ids = logits.argmax(dim=-1).tolist()
        keep_local_indices = []
        next_input = []
        for local_index, token_id in enumerate(token_ids):
            index = active_batch_indices[local_index]
            if token_id == tokenizer.eos_token_id:
                stopped[index] = True
                continue
            generated[index].append(int(token_id))
            counts[index] += 1
            if token_id in line_terminator_ids:
                stopped[index] = True
                continue
            if offset + 1 >= rooms[local_index]:
                continue
            keep_local_indices.append(local_index)
            next_input.append(int(token_id))
        if not keep_local_indices:
            break
        if len(keep_local_indices) < len(active_batch_indices):
            cache_indices = torch.tensor(keep_local_indices, dtype=torch.long, device=device)
            past = _select_cache_rows(past, cache_indices)
            attention_mask = attention_mask.index_select(0, cache_indices)
            rooms = [rooms[index] for index in keep_local_indices]
        active_batch_indices = [active_batch_indices[index] for index in keep_local_indices]
        token = torch.tensor(next_input, dtype=torch.long, device=device).unsqueeze(1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((len(next_input), 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        position_ids = attention_mask.cumsum(dim=-1)[:, -1:].sub(1).clamp_min(0)
        output = model(
            input_ids=token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
    for index in active_indices:
        text = tokenizer.decode(
            generated[index], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        answer = text.splitlines()[0] if text else ""
        results[index] = (answer, counts[index], not stopped[index])
    return [result for result in results if result is not None]


def _generate_qa(model: Any, tokenizer: Any, question: str, *, context: int, cap: int) -> tuple[str, int, bool]:
    return _generate_qa_batch(model, tokenizer, [question], context=context, cap=cap)[0]


def _generate_reasoning_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    context: int,
    cap: int,
    rollout: Any,
) -> tuple[Any, ...]:
    items = [
        (tokenizer.encode(f"Q: {row['question']}\n", add_special_tokens=False), row)
        for row in rows
    ]
    return generate_native_rollout_batch(
        model,
        tokenizer,
        items,
        context_length=context,
        max_completion_tokens=cap,
        temperature=0.0,
        rollout=rollout,
        greedy=True,
    )


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


def _progress_identity(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    identity_path = path.with_suffix(path.suffix + ".identity.json")
    if path.exists() and not identity_path.is_file():
        raise ValueError(f"Evaluation progress is missing its identity file: {path}")
    if identity_path.is_file():
        saved = json.loads(identity_path.read_text(encoding="utf-8"))
        if saved != identity:
            raise ValueError(f"Evaluation progress identity changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{identity_path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(identity, handle, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, identity_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if not path.is_file():
        return []
    content = path.read_bytes()
    complete_content = content
    if content and not content.endswith(b"\n"):
        last_newline = content.rfind(b"\n")
        complete_content = content[: last_newline + 1] if last_newline >= 0 else b""
        # Discard a single torn append so the next resumed row starts cleanly.
        with path.open("r+b") as handle:
            handle.truncate(len(complete_content))
    records = []
    for line_number, raw_line in enumerate(complete_content.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            records.append(json.loads(raw_line))
        except json.JSONDecodeError:
            raise ValueError(f"Corrupt evaluation progress row {line_number}: {path}") from None
    return records


def _append_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_progress_rows(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    if len(records) > len(rows):
        raise ValueError("Evaluation progress contains more predictions than fixed questions")
    for index, record in enumerate(records):
        row = rows[index]
        if (
            int(record.get("row_index", -1)) != index
            or record.get("module") != row.get("module")
            or record.get("question") != row.get("question")
            or record.get("target") != row.get("answer")
        ):
            raise ValueError(f"Evaluation progress question identity changed at row {index}")


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
            progress_path = output_dir / "progress" / f"{split_name}_{mode}.jsonl"
            progress_identity = {
                "schema": "exp16_hf_eval_progress_v1",
                "checkpoint_sha256": checkpoint_identity.get("checkpoint_sha256"),
                "base_model_revision": str(config.model.revision) if checkpoint is None else None,
                "token_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
                "split": split_name,
                "split_identity_sha256": expected["fixed_question_identity_sha256"],
                "format": mode,
                "rows": count,
                "completion_cap": int(config.eval.max_new_tokens),
                "rollout": dict(config.rollout),
            }
            records = _progress_identity(progress_path, progress_identity)
            _validate_progress_rows(records, rows)
            start_index = len(records)
            if mode == "ordinary_qa":
                batch_size = max(1, int(config.eval.get("batch_size", 1)))
                for batch_start in range(start_index, len(rows), batch_size):
                    batch_rows = rows[batch_start:batch_start + batch_size]
                    predictions = _generate_qa_batch(
                        model, tokenizer, [str(row["question"]) for row in batch_rows],
                        context=int(config.model.context_length), cap=int(config.eval.max_new_tokens),
                    )
                    for offset, (row, prediction_result) in enumerate(zip(batch_rows, predictions, strict=True)):
                        index = batch_start + offset
                        prediction, token_count, capped = prediction_result
                        digit_width, operator_count = question_features(str(row["question"]))
                        base = {
                            "row_index": index, "module": row["module"], "question": row["question"],
                            "target": row["answer"], "digit_width": digit_width,
                            "operator_count": operator_count, "policy_tokens": 0, "executor_tokens": 0,
                            "call_count": 0, "executor_failure": False, "trace_valid": False,
                            "full_success": False,
                        }
                        correct, valid = verify_answer(
                            prediction, str(row["answer"]), question=str(row["question"])
                        )
                        record = base | {
                            "prediction": prediction, "correct": bool(correct and valid),
                            "format_valid": valid, "capped": capped, "output_tokens": token_count,
                            "policy_tokens": token_count,
                            "termination_reason": "budget_exceeded" if capped else "newline_or_eos",
                        }
                        records.append(record)
                        _append_progress(progress_path, record)
            else:
                batch_size = max(1, int(config.eval.get("batch_size", 1)))
                for batch_start in range(start_index, len(rows), batch_size):
                    batch_rows = rows[batch_start : batch_start + batch_size]
                    rollouts = _generate_reasoning_batch(
                        model,
                        tokenizer,
                        batch_rows,
                        context=int(config.model.context_length),
                        cap=int(config.eval.max_new_tokens),
                        rollout=config.rollout,
                    )
                    for offset, (row, rollout_result) in enumerate(
                        zip(batch_rows, rollouts, strict=True)
                    ):
                        index = batch_start + offset
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
                        completion_ids = rollout_result.token_ids
                        if completion_ids and completion_ids[-1] == tokenizer.eos_token_id:
                            completion_ids = completion_ids[:-1]
                        text = tokenizer.decode(
                            completion_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        scored = score_native_reasoning(
                            text,
                            row,
                            capped=rollout_result.termination_reason == "budget_exceeded",
                            failure=(
                                rollout_result.termination_reason
                                if rollout_result.termination_reason not in {"eos", "budget_exceeded"}
                                else None
                            ),
                            format_weight=float(config.rlvr.reward_format_weight),
                            success_weight=float(config.rlvr.reward_success_weight),
                            max_calls=int(config.rollout.max_calls),
                            max_scalar_chars=int(config.rollout.max_scalar_chars),
                        )
                        record = base | {
                            "prediction": text,
                            "correct": scored.answer_correct,
                            "format_valid": scored.format_valid,
                            "trace_valid": scored.trace_valid,
                            "full_success": scored.full_success,
                            "capped": rollout_result.termination_reason == "budget_exceeded",
                            "output_tokens": len(rollout_result.token_ids),
                            "call_count": scored.call_count,
                            "executor_failure": rollout_result.termination_reason not in {"eos", "budget_exceeded"},
                            "policy_tokens": rollout_result.policy_tokens,
                            "executor_tokens": rollout_result.executor_tokens,
                            "termination_reason": rollout_result.termination_reason,
                        }
                        records.append(record)
                        _append_progress(progress_path, record)
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
