import json
import os

from tinymath.data.browser import discover_bundles, read_jsonl


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
