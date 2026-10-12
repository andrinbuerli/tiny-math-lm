from __future__ import annotations

from tinymath.data.char_tokenizer import CharacterTokenizer


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
