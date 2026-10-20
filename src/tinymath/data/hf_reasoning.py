"""Data identity and native-token formatting for the Qwen3 reasoning protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tinymath.data.load import SELECTED_MODULES

PROMPT_FORMAT = "Q: {question}\n"
COMPLETION_FORMAT = "<think>\n{scratchpad}\n</think>\nA: {answer}"
FORMAT_VERSION = "exp16-qwen3-reasoning-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_source_bundle(
    bundle_dir: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_seed: int = 18,
) -> dict[str, Any]:
    """Fail closed unless the exact, previously verified reasoning-v3 bundle is used."""
    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reasoning-v3 manifest not found: {manifest_path}")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_sha256:
        raise ValueError(
            "Reasoning-v3 source manifest does not match Experiment 15 provenance "
            f"({manifest_hash} != {expected_manifest_sha256})"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("seed", -1)) != expected_seed:
        raise ValueError("Reasoning-v3 source bundle seed does not match Experiment 16")
    modules = set(manifest.get("selected_modules", []))
    if modules != set(SELECTED_MODULES):
        raise ValueError(f"Reasoning-v3 bundle has unexpected module inventory: {sorted(modules)}")
    if manifest.get("trace_protocol") != "reasoning_v3":
        raise ValueError("Experiment 16 requires the verified reasoning-v3 annotation protocol")
    return manifest


def select_training_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain every parent and only standalone subtasks marked eligible."""
    selected = []
    for row in rows:
        if row.get("is_sub_task", False):
            if row.get("training_eligible") is True:
                if row.get("overlapping_splits"):
                    raise ValueError("Eligible standalone subtask has a cross-pool overlap")
                selected.append(row)
        else:
            selected.append(row)
    if any(set(row.get("module", "") for row in selected) - set(SELECTED_MODULES)):
        raise ValueError("SFT rows contain a module outside the required five-module scope")
    return selected


def select_parent_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluation and prompt pools are parent-only; subtasks never enter evaluation."""
    selected = [row for row in rows if not row.get("is_sub_task", False)]
    if any(row.get("module") not in SELECTED_MODULES for row in selected):
        raise ValueError("Evaluation rows contain a module outside the required five-module scope")
    return selected


def identity_hash(rows: Iterable[dict[str, Any]]) -> tuple[str, int]:
    """Order-independent hash compatible with the Experiment 15 evaluation identity."""
    identities = [
        json.dumps(
            [row.get("module", ""), row.get("question", ""), row.get("answer", "")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for row in rows
    ]
    digest = hashlib.sha256()
    for identity in sorted(identities):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), len(identities)


def render_prompt(question: str) -> str:
    """Render the native-tokenizer prompt without rewriting original question text."""
    return PROMPT_FORMAT.format(question=question)


def render_completion(scratchpad: str, answer: str) -> str:
    """Render only the structural scratchpad boundaries with Qwen's native markers."""
    if not scratchpad.strip() or not answer or answer != answer.strip():
        raise ValueError("Reasoning completions require a scratchpad and scalar answer")
    if "\n" in answer or "\r" in answer:
        raise ValueError("Reasoning answer must occupy one line")
    return COMPLETION_FORMAT.format(scratchpad=scratchpad, answer=answer)


def encode_completion(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    context_length: int = 512,
    completion_cap: int = 512,
) -> tuple[list[int], list[int], int]:
    """Return input IDs, completion-only labels, and prompt length, including true EOS."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Qwen tokenizer must define a native EOS token")
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_id is None or end_think_id is None or think_id == tokenizer.unk_token_id:
        raise ValueError("Qwen native tokenizer is missing its existing thinking-token IDs")
    if tokenizer.encode("<think>", add_special_tokens=False) != [think_id]:
        raise ValueError("<think> is not one native tokenizer token")
    if tokenizer.encode("</think>", add_special_tokens=False) != [end_think_id]:
        raise ValueError("</think> is not one native tokenizer token")

    prompt = tokenizer.encode(render_prompt(str(row["question"])), add_special_tokens=False)
    completion_text = render_completion(str(row["scratchpad"]), str(row["answer"]))
    completion = tokenizer.encode(completion_text, add_special_tokens=False) + [int(eos_id)]
    if len(completion) > completion_cap:
        raise ValueError(
            f"Completion exceeds configured cap: {len(completion)} > {completion_cap}"
        )
    if len(prompt) + len(completion) > context_length:
        raise ValueError(
            "Complete eligible row exceeds context; refusing to truncate or drop it "
            f"(prompt={len(prompt)}, completion={len(completion)}, context={context_length})"
        )
    ids = [int(token) for token in (*prompt, *completion)]
    labels = [-100] * len(prompt) + [int(token) for token in completion]
    return ids, labels, len(prompt)
