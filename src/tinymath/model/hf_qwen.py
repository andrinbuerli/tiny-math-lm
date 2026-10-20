"""Pinned Hugging Face loading helpers for the Experiment 16 Qwen3 Base model."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinymath.data.hf_reasoning import sha256_file
from tinymath.data.hf_token_cache import native_tokenizer_identity


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"Unsupported Hugging Face dtype: {name}") from None


def load_tokenizer(config: Any) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.repo_id),
        revision=str(config.revision),
        cache_dir=str(config.cache_dir),
        trust_remote_code=bool(config.trust_remote_code),
        use_fast=True,
    )
    # PAD intentionally aliases native EOS; attention and label masks distinguish them.
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("Experiment 16 requires the tokenizer's native PAD/EOS sharing")
    native_tokenizer_identity(tokenizer)
    return tokenizer


def load_model(config: Any, *, checkpoint: str | Path | None = None, training: bool = False):
    source = str(checkpoint) if checkpoint is not None else str(config.repo_id)
    revision = None if checkpoint is not None else str(config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        cache_dir=str(config.cache_dir),
        torch_dtype=_dtype(str(config.torch_dtype)),
        attn_implementation=str(config.attn_implementation),
        trust_remote_code=bool(config.trust_remote_code),
        low_cpu_mem_usage=True,
    )
    if len(model.get_input_embeddings().weight) < int(model.config.vocab_size):
        raise ValueError("Loaded model input embedding table is smaller than its declared vocabulary")
    if training:
        model.config.use_cache = False
        if bool(config.gradient_checkpointing):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.config.use_cache = bool(config.use_cache)
    return model


def model_provenance(config: Any, tokenizer: Any, model: Any) -> dict[str, Any]:
    config_path = hf_hub_download(
        repo_id=str(config.repo_id),
        filename="config.json",
        revision=str(config.revision),
        cache_dir=str(config.cache_dir),
    )
    tokenizer_hash, tokenizer_identity = native_tokenizer_identity(tokenizer)
    return {
        "model_repo_id": str(config.repo_id),
        "model_revision": str(config.revision),
        "model_config_sha256": sha256_file(config_path),
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_identity": tokenizer_identity,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": model.config.to_dict(),
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "huggingface-hub")
        },
    }


def save_model_identity(path: str | Path, identity: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(identity, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def completion_loss(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Causal completion-only cross-entropy; true EOS labels survive PAD/EOS sharing."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    shifted_logits = outputs.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels != -100
    token_count = int(valid.sum().item())
    if token_count == 0:
        raise ValueError("Batch contains no supervised completion tokens")
    loss_sum = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, token_count
