import json
import os

from tinymath.data.browser import count_example_tokens, discover_bundles, read_jsonl
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt


def test_discover_bundles_returns_manifest_directories(tmp_path):
    older = tmp_path / "processed" / "older"
    newer = tmp_path / "processed" / "newer"
    older.mkdir(parents=True)
    newer.mkdir()
    (older / "manifest.json").write_text("{}", encoding="utf-8")
    (newer / "manifest.json").write_text("{}", encoding="utf-8")
    os.utime(older / "manifest.json", ns=(10_000_000_000, 10_000_000_000))
    (tmp_path / "processed" / "incomplete").mkdir()

    assert discover_bundles(tmp_path) == [newer, older]


def test_read_jsonl_skips_blank_lines_and_preserves_unicode(tmp_path):
    path = tmp_path / "test.jsonl"
    row = {"question": "2 + 2 = ?", "answer": "4", "module": "arithmetic__add_or_sub"}
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")

    assert read_jsonl(path) == [row]


def test_count_example_tokens_matches_reasoning_model_segments():
    row = {
        "question": "2 + 2",
        "scratchpad": "step:+(2,2)\n=>4",
        "answer": "4",
    }
    tokenizer = CharacterTokenizer.from_texts(
        [format_prompt(row["question"], protocol="reasoning_v2"), row["scratchpad"], "\nA: 4"],
        protocol="reasoning_v2",
    )

    counts = count_example_tokens(row, tokenizer)

    assert counts["prompt"] == len(
        tokenizer.encode(format_prompt(row["question"], protocol="reasoning_v2"), add_bos=True)
    )
    assert counts["scratchpad"] == len(tokenizer.encode("\n" + row["scratchpad"] + "\n")) + 2
    assert counts["answer"] == len(tokenizer.encode("\nA: 4")) + 1


def test_count_example_tokens_answer_only_has_no_scratchpad_segment():
    row = {"question": "2 + 2", "answer": "4"}
    tokenizer = CharacterTokenizer.from_texts(
        [format_prompt(row["question"]), row["answer"]]
    )

    counts = count_example_tokens(row, tokenizer)

    assert counts["prompt"] == len(tokenizer.encode(format_prompt(row["question"]), add_bos=True))
    assert counts["scratchpad"] == 0
    assert counts["answer"] == len(tokenizer.encode("4")) + 1
