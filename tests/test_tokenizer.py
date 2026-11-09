from __future__ import annotations

from tiny_math_lm.data.char_tokenizer import CharacterTokenizer


def test_each_unicode_codepoint_is_one_token_and_round_trips():
    tokenizer = CharacterTokenizer.from_texts(["Q: 23 + 48.\nA: 71", "−π"])
    text = "Q: 23 + 48.\nA: 71"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert len(ids) == len(text) + 2
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(ids, skip_special=False).startswith("<BOS>")


def test_unknown_characters_are_errors_unless_explicitly_allowed():
    tokenizer = CharacterTokenizer.from_texts(["12"])
    try:
        tokenizer.encode("3")
    except ValueError as error:
        assert "Unknown character U+0033" in str(error)
    else:
        raise AssertionError("unknown character should fail")
    assert tokenizer.encode("3", allow_unk=True) == [tokenizer.unk_id]


def test_batch_padding_and_tokenizer_persistence(workspace_tmpdir):
    tokenizer = CharacterTokenizer.from_texts(["a", "abc"])
    ids, mask = tokenizer.batch(["a", "abc"])
    assert ids.shape == mask.shape == (2, 3)
    assert mask.tolist() == [[True, False, False], [True, True, True]]
    path = workspace_tmpdir / "tokenizer.json"
    tokenizer.save(path)
    restored = CharacterTokenizer.load(path)
    assert restored.sha256 == tokenizer.sha256
    assert restored.encode("abc") == tokenizer.encode("abc")


def test_reasoning_tokenizer_has_six_reserved_ids_and_keeps_markers_visible(workspace_tmpdir):
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: Calculate 47 + 28.\n", "+(47,28)\n7,8,0:5,1\nA: 75"], protocol="reasoning_v1"
    )
    assert [
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.unk_id,
        tokenizer.scp_id,
        tokenizer.scp_end_id,
    ] == [0, 1, 2, 3, 4, 5]
    prompt = "Q: Calculate 47 + 28.\n"
    completion = tokenizer.encode_completion("75", scratchpad="+(47,28)\n7,8,0:5,1")
    assert tokenizer.encode(prompt, add_bos=True)[0] == tokenizer.bos_id
    visible = tokenizer.decode(completion)
    assert visible.startswith("<scp>\n+(47,28)")
    assert "\n</scp>\nA: 75" in visible
    path = workspace_tmpdir / "reasoning-tokenizer.json"
    tokenizer.save(path)
    restored = CharacterTokenizer.load(path)
    assert restored.protocol == "reasoning_v1"
    assert restored.sha256 == tokenizer.sha256
    assert restored.scp_id == 4 and restored.scp_end_id == 5


def test_marker_text_in_question_is_encoded_as_ordinary_characters():
    tokenizer = CharacterTokenizer.from_texts(
        ["Q: write <scp> literally\n", "trace\nA: 1"], protocol="reasoning_v1"
    )
    question_ids = tokenizer.encode("Q: write <scp> literally\n")
    assert tokenizer.scp_id not in question_ids
    assert tokenizer.scp_end_id not in question_ids
