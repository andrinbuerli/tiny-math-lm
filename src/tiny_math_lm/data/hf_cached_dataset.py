"""Memory-mapped reader and batching helpers for native Hugging Face token caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class HFTokenSplit:
    def __init__(self, cache_dir: str | Path, split: str):
        self.cache_dir = Path(cache_dir)
        self.split = split
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self.metadata = manifest["splits"][split]
        count = int(self.metadata["rows"])
        self.offsets = np.memmap(
            self.cache_dir / f"{split}.offsets.u64le", dtype="<u8", mode="r", shape=(count + 1,)
        )
        self.prompt_lengths = np.memmap(
            self.cache_dir / f"{split}.prompt_lengths.u32le", dtype="<u4", mode="r", shape=(count,)
        )
        token_count = (self.cache_dir / f"{split}.ids.u32le").stat().st_size // 4
        self.token_ids = np.memmap(
            self.cache_dir / f"{split}.ids.u32le", dtype="<u4", mode="r", shape=(token_count,)
        )
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != token_count:
            raise ValueError(f"Corrupt native-token cache offsets for {split}")

    def __len__(self) -> int:
        return int(self.prompt_lengths.shape[0])

    def get(self, index: int) -> tuple[list[int], int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.token_ids[start:end].astype(np.int64).tolist(), int(self.prompt_lengths[index])

    def batch(
        self,
        indices: list[int],
        *,
        pad_id: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not indices:
            raise ValueError("Cannot form an empty native-token batch")
        sequences = [self.get(index) for index in indices]
        width = max(len(ids) for ids, _prompt_length in sequences)
        input_ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(indices), width), dtype=torch.long)
        labels = torch.full((len(indices), width), -100, dtype=torch.long)
        for row_index, (ids, prompt_length) in enumerate(sequences):
            length = len(ids)
            input_ids[row_index, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row_index, :length] = 1
            labels[row_index, prompt_length:length] = torch.tensor(
                ids[prompt_length:], dtype=torch.long
            )
        return (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

    def rows(self):
        path = self.cache_dir / f"{self.split}.rows.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
