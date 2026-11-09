"""Compact decoder-only Transformer using RoPE, RMSNorm, SDPA and SwiGLU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 128
    d_ff: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @classmethod
    def from_mapping(cls, values: Any) -> GPTConfig:
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: values[key] for key in fields if key in values})


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    frequencies = theta ** (
        -torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    if positions.ndim == 1:
        cosine = angles.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sine = angles.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    elif positions.ndim == 2:
        cosine = angles.cos().to(dtype=x.dtype).unsqueeze(1)
        sine = angles.sin().to(dtype=x.dtype).unsqueeze(1)
    else:
        raise ValueError("RoPE positions must be [sequence] or [batch, sequence]")
    return x * cosine + _rotate_half(x) * sine


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.theta = config.rope_theta
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        positions = torch.arange(length, device=x.device)
        q, k = _apply_rope(q, positions, self.theta), _apply_rope(k, positions, self.theta)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(attended.transpose(1, 2).contiguous().view(batch, length, width))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run attention incrementally, returning the concatenated KV cache."""
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        past_length = past_key_value[0].shape[-2] if past_key_value is not None else 0
        positions = (torch.arange(past_length, past_length + length, device=x.device)
                     if position_ids is None else position_ids)
        q, k = _apply_rope(q, positions, self.theta), _apply_rope(k, positions, self.theta)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        # Cached calls only contain the causal prefix and current token. A
        # noncausal kernel is correct there and avoids SDPA's upper-left mask
        # alignment when query and key lengths differ.
        if attention_mask is None:
            attended = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=past_key_value is None,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            expected_shape = (batch, past_length + length)
            if attention_mask.shape != expected_shape or attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean [batch, cached_sequence]")
            if past_key_value is None:
                causal = torch.ones(
                    length, length, dtype=torch.bool, device=x.device
                ).tril()
                attention_mask = attention_mask[:, None, None, :] & causal[None, None, :, :]
            else:
                attention_mask = attention_mask[:, None, None, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        result = self.out(attended.transpose(1, 2).contiguous().view(batch, length, width))
        return result, (k, v)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.value = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, present = self.attn.forward_with_cache(
            self.attn_norm(x), past_key_value, attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = x + attended
        return x + self.ffn(self.ffn_norm(x)), present


class TinyMathGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.context_length:
            raise ValueError("input_ids must be [batch, sequence] within configured context length")
        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x)
        return F.linear(self.final_norm(x), self.token_embedding.weight)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Return logits and per-layer KV state for prefix-cached decoding."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if position_ids is None and past_length + input_ids.shape[1] > self.config.context_length:
            raise ValueError("cached input exceeds configured context length")
        if position_ids is not None and (
            position_ids.shape != input_ids.shape
            or position_ids.dtype not in {torch.int32, torch.int64}
            or torch.any(position_ids < 0)
            or torch.any(position_ids >= self.config.context_length)
        ):
            raise ValueError("position_ids must be integer [batch, sequence] within context")
        if attention_mask is not None and attention_mask.shape != (
            input_ids.shape[0], past_length + input_ids.shape[1]
        ):
            raise ValueError("attention_mask must cover the cached prefix and current input")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("KV cache layer count does not match the model")
        x = self.dropout(self.token_embedding(input_ids))
        present: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            cached = None if past_key_values is None else past_key_values[index]
            x, layer_present = block.forward_with_cache(
                x, cached, attention_mask=attention_mask, position_ids=position_ids
            )
            present.append(layer_present)
        return F.linear(self.final_norm(x), self.token_embedding.weight), tuple(present)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        was_training = self.training
        self.eval()
        output = input_ids
        cacheable = output.shape[1] + max_new_tokens <= self.config.context_length
        cache = None
        for step in range(max_new_tokens):
            if cacheable:
                token_input = input_ids if step == 0 else output[:, -1:]
                logits, cache = self.forward_with_cache(token_input, cache)
                logits = logits[:, -1]
            else:
                context = output[:, -self.config.context_length :]
                logits = self(context)[:, -1]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            output = torch.cat((output, next_id), dim=1)
            if eos_id is not None and torch.all(next_id == eos_id):
                break
        self.train(was_training)
        return output


def parameter_report(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    """Instantiate configs and report actual count against each target."""
    report = {}
    for name, values in configs.items():
        model = TinyMathGPT(GPTConfig.from_mapping(values))
        count = model.count_parameters()
        target = int(values.get("target_params", count))
        report[name] = {
            "parameters": count,
            "target": target,
            "relative_error": abs(count - target) / target,
        }
    return report
