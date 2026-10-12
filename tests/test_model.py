from __future__ import annotations

import torch

from tinymath.model.gpt import GPTConfig, TinyMathGPT


def test_model_forward_generation_and_tied_embeddings():
    config = GPTConfig(vocab_size=19, context_length=16, n_layers=2, n_heads=2, d_model=16, d_ff=32)
    model = TinyMathGPT(config)
    ids = torch.randint(0, config.vocab_size, (2, 7))
    logits = model(ids)
    assert logits.shape == (2, 7, config.vocab_size)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    generated = model.generate(ids[:, :3], max_new_tokens=2)
    assert generated.shape == (2, 5)
    assert model.count_parameters() == sum(parameter.numel() for parameter in model.parameters())


def test_model_rejects_context_overflow_and_invalid_head_width():
    model = TinyMathGPT(GPTConfig(vocab_size=9, context_length=3, n_layers=1, n_heads=1, d_model=8, d_ff=16))
    try:
        model(torch.ones((1, 4), dtype=torch.long))
    except ValueError as error:
        assert "context length" in str(error)
    else:
        raise AssertionError("context overflow should fail")
    try:
        TinyMathGPT(GPTConfig(vocab_size=9, n_layers=1, n_heads=1, d_model=7, d_ff=12))
    except ValueError as error:
        assert "even attention head" in str(error)
    else:
        raise AssertionError("odd head width should fail")
