"""Strict Unicode-code-point tokenizers for answer-only and reasoning protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from tinymath.config import WORKSPACE

LEGACY_SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
REASONING_SPECIAL_TOKENS = (*LEGACY_SPECIAL_TOKENS, "<scp>", "</scp>")
TOKENIZER_VERSION = 2


class CharacterTokenizer:
    """Map each ordinary Unicode code point to one stable token ID."""

    def __init__(self, characters: Iterable[str], *, protocol: str = "answer_only"):
        if protocol not in {"answer_only", "reasoning_v1", "reasoning_v2"}:
            raise ValueError(f"Unsupported tokenizer protocol: {protocol}")
        self.protocol = protocol
        self.special_tokens = (
            LEGACY_SPECIAL_TOKENS if protocol == "answer_only" else REASONING_SPECIAL_TOKENS
        )
        unique = set(characters)
        if any(len(character) != 1 for character in unique):
            raise ValueError("Ordinary tokens must be exactly one Unicode code point")
        self.id_to_token = list(self.special_tokens) + sorted(unique - set(self.special_tokens))
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        payload = json.dumps(self.id_to_token, ensure_ascii=False, separators=(",", ":"))
        self.sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_texts(
        cls, texts: Iterable[str], *, protocol: str = "answer_only"
    ) -> CharacterTokenizer:
        return cls((character for text in texts for character in text), protocol=protocol)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def scp_id(self) -> int:
        if self.protocol not in {"reasoning_v1", "reasoning_v2"}:
            raise ValueError("<scp> is available only in a reasoning tokenizer")
        return 4

    @property
    def scp_end_id(self) -> int:
        if self.protocol not in {"reasoning_v1", "reasoning_v2"}:
            raise ValueError("</scp> is available only in a reasoning tokenizer")
        return 5

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False, allow_unk: bool = False
    ) -> list[int]:
        ids = []
        for character in text:
            token_id = self.token_to_id.get(character)
            if token_id is None:
                if not allow_unk:
                    raise ValueError(f"Unknown character U+{ord(character):04X}: {character!r}")
                token_id = self.unk_id
            ids.append(token_id)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_completion(self, answer: str, *, scratchpad: str | None = None) -> list[int]:
        """Encode the completion with protocol markers inserted as reserved IDs.

        Marker insertion is explicit here; question text is always encoded as ordinary
        Unicode characters and is never searched or rewritten for marker substrings.
        """
        if self.protocol == "answer_only":
            if scratchpad is not None:
                raise ValueError("Legacy answer-only completions cannot contain scratchpads")
            return self.encode(answer, add_eos=True)
        if not answer.strip() or not scratchpad or not scratchpad.strip():
            raise ValueError("Reasoning completions require a nonempty scratchpad and answer")
        return (
            [self.scp_id]
            + self.encode("\n" + scratchpad + "\n")
            + [self.scp_end_id]
            + self.encode("\nA: " + answer)
            + [self.eos_id]
        )

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        result = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of vocabulary: {token_id}")
            token = self.id_to_token[token_id]
            if token_id < len(self.special_tokens):
                # Reasoning boundaries remain visible for qualitative inspection.
                if not skip_special or token_id in (4, 5):
                    result.append(token)
                continue
            result.append(token)
        return "".join(result)

    def batch(
        self, texts: Sequence[str], *, add_bos: bool = False, add_eos: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad encoded strings and return IDs plus a boolean nonpadding mask."""
        rows = [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
        width = max((len(row) for row in rows), default=0)
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, row in enumerate(rows):
            if row:
                ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
                mask[index, : len(row)] = True
        return ids, mask

    def save(self, path: str | Path) -> None:
        destination = Path(path).resolve()
        if destination != WORKSPACE and WORKSPACE not in destination.parents:
            raise ValueError(f"Tokenizer files must be saved inside {WORKSPACE}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": (
                3
                if self.protocol == "reasoning_v2"
                else TOKENIZER_VERSION
                if self.protocol == "reasoning_v1"
                else 1
            ),
            "protocol": self.protocol,
            "special_tokens": list(self.special_tokens),
            "id_to_token": self.id_to_token,
            "token_to_id": self.token_to_id,
            "sha256": self.sha256,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        protocol = "answer_only" if version == 1 else payload.get("protocol")
        expected = LEGACY_SPECIAL_TOKENS if protocol == "answer_only" else REASONING_SPECIAL_TOKENS
        if version not in (1, TOKENIZER_VERSION, 3) or payload.get("special_tokens") != list(expected):
            raise ValueError("Unsupported tokenizer format")
        tokenizer = cls(payload["id_to_token"][len(expected) :], protocol=protocol)
        if tokenizer.id_to_token != payload["id_to_token"] or tokenizer.sha256 != payload["sha256"]:
            raise ValueError("Tokenizer vocabulary or digest is inconsistent")
        return tokenizer
