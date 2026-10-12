from __future__ import annotations

import torch

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.train.sft import build_completion_batch, completion_only_loss


def test_loss_mask_begins_at_first_answer_target_and_includes_eos():
    examples = [{"prompt": "Q: 1+1.\nA: ", "completion": "2"}]
    tokenizer = CharacterTokenizer.from_texts([examples[0]["prompt"] + "2"])
    _inputs, targets, mask = build_completion_batch(examples, tokenizer, context_length=32)
    prompt_length = len(tokenizer.encode(examples[0]["prompt"], add_bos=True))
    assert not mask[0, : prompt_length - 1].any()
    assert mask[0, prompt_length - 1 :].tolist() == [True, True]
    assert targets[0, -2].item() == tokenizer.token_to_id["2"]
    assert targets[0, -1].item() == tokenizer.eos_id


def test_completion_loss_ignores_prompt_and_padding_logits():
    examples = [
        {"prompt": "Q: 1\nA: ", "completion": "2"},
        {"prompt": "Q: 2\nA: ", "completion": "30"},
    ]
    tokenizer = CharacterTokenizer.from_texts([e["prompt"] + e["completion"] for e in examples])
    _inputs, targets, mask = build_completion_batch(examples, tokenizer, context_length=32)
    logits = torch.zeros((*targets.shape, tokenizer.vocab_size))
    loss = completion_only_loss(logits, targets, mask)
    logits[~mask] = 10000
    assert torch.isfinite(loss)
    assert torch.equal(loss, completion_only_loss(logits, targets, mask))


def test_long_example_is_rejected_without_truncation():
    tokenizer = CharacterTokenizer.from_texts(["Q: long\nA: answer"])
    try:
        build_completion_batch(
            [{"prompt": "Q: long\nA: ", "completion": "answer"}], tokenizer, context_length=4
        )
    except ValueError as error:
        assert "never silently truncated" in str(error)
    else:
        raise AssertionError("overlength examples must be excluded or rejected explicitly")
