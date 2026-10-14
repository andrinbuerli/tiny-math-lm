"""Completion-only SFT batch construction and loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from tinymath.data.char_tokenizer import CharacterTokenizer


def build_completion_batch(
    examples: Sequence[dict[str, str]], tokenizer: CharacterTokenizer, *, context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build `[BOS prompt completion EOS]`; mask prompt and padding targets."""
    if not examples:
        raise ValueError("Cannot build a training batch with no examples")
    encoded: list[tuple[list[int], int]] = []
    for example in examples:
        prompt = example["prompt"]
        completion = example["completion"]
        if not prompt or not completion:
            raise ValueError("Each training example needs a nonempty prompt and completion")
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        completion_ids = tokenizer.encode_completion(
            completion, scratchpad=example.get("scratchpad")
        )
        row = prompt_ids + completion_ids
        if len(row) > context_length:
            raise ValueError(
                f"Example requires {len(row)} tokens, above context length {context_length}; "
                "examples are never silently truncated"
            )
        encoded.append((row, len(prompt_ids)))

    width = max((len(row) for row, _ in encoded), default=0)
    inputs = torch.full((len(encoded), width - 1), tokenizer.pad_id, dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
    for index, (row, prompt_size) in enumerate(encoded):
        if len(row) < 2:
            raise ValueError("Each training example must contain a prompt and completion")
        shifted_inputs = torch.tensor(row[:-1], dtype=torch.long)
        shifted_targets = torch.tensor(row[1:], dtype=torch.long)
        inputs[index, : len(shifted_inputs)] = shifted_inputs
        targets[index, : len(shifted_targets)] = shifted_targets
        # target index j predicts token row[j + 1]; completion starts at prompt_size.
        loss_mask[index, prompt_size - 1 : len(shifted_targets)] = True
    return inputs, targets, loss_mask


def completion_only_loss(
    logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """Cross entropy averaged over completion characters and EOS only."""
    if logits.shape[:2] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError("logits, targets and loss_mask sequence dimensions must match")
    if loss_mask.dtype != torch.bool or not torch.any(loss_mask):
        raise ValueError("loss_mask must select at least one completion target")
    return F.cross_entropy(logits[loss_mask].float(), targets[loss_mask])
