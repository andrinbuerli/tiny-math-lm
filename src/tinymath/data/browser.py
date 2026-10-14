"""Read-only helpers for browsing prepared data bundles."""

from __future__ import annotations

import json
from pathlib import Path

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt


def discover_bundles(data_dir: Path) -> list[Path]:
    """Return prepared data directories that contain a readable manifest."""
    if not data_dir.exists():
        return []
    manifests = list(data_dir.glob("processed/*/manifest.json"))
    manifests.sort(
        key=lambda manifest: (manifest.stat().st_mtime_ns, manifest.parent.name), reverse=True
    )
    return [manifest.parent for manifest in manifests]


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Read a prepared split, ignoring blank lines and reporting malformed rows."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object on line {line_number} of {path}")
            rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def count_example_tokens(
    row: dict[str, str], tokenizer: CharacterTokenizer
) -> dict[str, int]:
    """Count model token IDs for the prompt, reasoning trace, and answer segments."""
    prompt = len(
        tokenizer.encode(
            format_prompt(row.get("question", ""), protocol=tokenizer.protocol), add_bos=True
        )
    )
    answer = row.get("answer", "")
    scratchpad = row.get("scratchpad", "")
    if tokenizer.protocol in {"reasoning_v1", "reasoning_v2"}:
        scratchpad_tokens = (
            len(tokenizer.encode("\n" + scratchpad + "\n")) + 2 if scratchpad else 0
        )
        answer_tokens = len(tokenizer.encode("\nA: " + answer)) + 1
    else:
        scratchpad_tokens = 0
        answer_tokens = len(tokenizer.encode(answer)) + 1
    return {
        "prompt": prompt,
        "scratchpad": scratchpad_tokens,
        "answer": answer_tokens,
    }
