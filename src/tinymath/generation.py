"""Cached character generation with optional exact-arithmetic observations."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.prompts import format_prompt
from tinymath.data.reasoning import CALL_POLICY, task_allows_calls
from tinymath.rewards.calls import EXECUTOR_VERSION, CallError, execute_call, validate_call
from tinymath.rewards.reasoning import SCORER_VERSION

ROLLOUT_VERSION = 3


@dataclass(frozen=True)
class RolloutConfig:
    executor: str = "none"
    max_calls: int = 32
    max_scalar_chars: int = 128
    format_weight: float = 0.1
    success_weight: float = 0.9
    scorer_version: int = SCORER_VERSION
    executor_version: int = EXECUTOR_VERSION
    rollout_version: int = ROLLOUT_VERSION
    token_accounting: str = "transcript_includes_executor_v1"
    delegation_policy: str = CALL_POLICY

    def __post_init__(self):
        if (self.scorer_version != SCORER_VERSION or self.executor_version != EXECUTOR_VERSION
                or self.rollout_version != ROLLOUT_VERSION
                or self.token_accounting != "transcript_includes_executor_v1"
                or self.delegation_policy != CALL_POLICY):
            raise ValueError("Unsupported rollout/scorer/executor/accounting version")
        if self.executor not in {"none", "exact"}:
            raise ValueError("executor must be none or exact")
        if self.max_calls < 1 or not 1 <= self.max_scalar_chars <= 1024:
            raise ValueError("Call limits must be positive; scalar limit must be <= 1024")
        if (not 0 <= self.format_weight < 1 or not 0 < self.success_weight <= 1
                or abs(self.format_weight + self.success_weight - 1) > 1e-9):
            raise ValueError("Nonnegative format and positive success weights must sum to 1")

    @classmethod
    def from_config(cls, config: Any = None) -> RolloutConfig:
        if isinstance(config, cls):
            return config
        if config is None:
            return cls()
        return cls(**dict(config))

    def identity(self) -> dict[str, Any]:
        return asdict(self)

    def scoring_kwargs(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in (
            "max_calls", "max_scalar_chars", "format_weight", "success_weight",
        )}


@dataclass(frozen=True)
class Rollouts:
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_log_probs: torch.Tensor
    entropy: torch.Tensor
    policy_mask: torch.Tensor | None = None
    termination_reasons: tuple[str, ...] = ()
    call_events: tuple[tuple[dict[str, Any], ...], ...] = ()

    @property
    def loss_mask(self) -> torch.Tensor:
        return self.completion_mask if self.policy_mask is None else self.policy_mask


class _CallState:
    def __init__(self, allowed: bool):
        self.allowed = allowed
        self.line = ""
        self.inside = False
        self.started = False
        self.dispatched = False
        self.pending: deque[int] = deque()
        self.events: list[dict[str, Any]] = []
        self.call_count = 0

    def observe(self, character: str, tokenizer: CharacterTokenizer, config: RolloutConfig,
                *, position: int, cap: int) -> None:
        if character == "\n":
            if not self.started and self.line == "<scp>":
                self.inside = True
                self.started = True
            elif self.line == "</scp>":
                self.inside = False
            elif self.inside and self.line.lstrip().startswith("call:"):
                if not self.allowed:
                    raise CallError("disallowed_call")
                if config.executor == "exact":
                    validate_call(self.line, max_scalar_chars=config.max_scalar_chars)
            self.line = ""
            self.dispatched = False
            return
        self.line += character
        if self.inside and self.line.lstrip() == "call:" and not self.allowed:
            raise CallError("disallowed_call")
        if (self.inside and self.line.startswith("call:") and not self.dispatched
                and self.line.endswith(" -> ")):
            if self.call_count >= config.max_calls:
                raise CallError("call_limit")
            if not self.allowed:
                raise CallError("disallowed_call")
            self.dispatched = True
            self.call_count += 1
            if config.executor == "none":
                return
            call = execute_call(self.line, max_scalar_chars=config.max_scalar_chars)
            try:
                result_ids = tokenizer.encode(call.result)
            except ValueError:
                raise CallError("unencodable_result") from None
            if len(result_ids) > cap - position - 1:
                raise CallError("budget_exceeded")
            self.pending.extend(result_ids)
            self.events.append({
                **asdict(call), "request_end": position + 1,
                "result_start": position + 1, "result_end": position + 1 + len(result_ids),
            })


@torch.no_grad()
def generate_rollouts(
    model: Any, prompt_ids: torch.Tensor, *, group_size: int = 1,
    max_completion_tokens: int, temperature: float, eos_id: int, pad_id: int,
    tokenizer: CharacterTokenizer | None = None, rollout: RolloutConfig | None = None,
    task_rows: list[dict[str, Any]] | None = None,
) -> Rollouts:
    """Generate equal-length prompt batches, with separate observation/action masks.

    Flattened lane order is prompt-major then group-major. The cap includes EOS
    and all executor observations; no prefix or context is ever truncated.
    """
    config = RolloutConfig.from_config(rollout)
    if prompt_ids.ndim != 2 or min(prompt_ids.shape) < 1:
        raise ValueError("prompts must be nonempty [batch, unpadded_sequence]")
    if group_size < 1 or max_completion_tokens < 1 or temperature < 0:
        raise ValueError("Invalid group size, completion cap, or temperature")
    if config.executor == "exact" and (tokenizer is None or tokenizer.protocol != "reasoning_v3"):
        raise ValueError("Exact call execution requires a reasoning_v3 tokenizer")
    batch, prompt_length = prompt_ids.shape
    cap = min(max_completion_tokens, model.config.context_length - prompt_length)
    if cap < 1:
        raise ValueError("Prompt leaves no room for a completion")
    lanes = batch * group_size
    sequences = prompt_ids[:, None, :].expand(batch, group_size, prompt_length).reshape(
        lanes, prompt_length
    )
    active = torch.ones(lanes, dtype=torch.bool, device=prompt_ids.device)
    if task_rows is not None and len(task_rows) != batch:
        raise ValueError("One task row is required per prompt")
    policies = []
    for index in range(batch):
        allowed = False
        if tokenizer is not None and tokenizer.protocol == "reasoning_v3":
            text = tokenizer.decode(prompt_ids[index].tolist())
            row = task_rows[index] if task_rows is not None else {
                "question": text[3:-1] if text.startswith("Q: ") and text.endswith("\n") else "",
            }
            if task_rows is not None and text != format_prompt(row["question"], protocol="reasoning_v3"):
                raise ValueError("Task metadata does not match the generation prompt")
            allowed = task_allows_calls(row)
        policies.extend([allowed] * group_size)
    states = [_CallState(allowed) for allowed in policies]
    reasons = ["budget_exceeded"] * lanes
    tokens, masks, policy_masks, old_logs, entropies = [], [], [], [], []
    cache = None
    was_training = model.training
    model.eval()
    try:
        for position in range(cap):
            logits, cache = model.forward_with_cache(
                sequences if cache is None else sequences[:, -1:], cache
            )
            valid = active.clone()
            forced = torch.tensor(
                [bool(state.pending) for state in states], device=prompt_ids.device
            ) & valid
            policy = valid & ~forced
            token = torch.full((lanes,), pad_id, device=prompt_ids.device, dtype=torch.long)
            log_values = torch.zeros(lanes, device=prompt_ids.device)
            entropy = torch.zeros_like(log_values)
            if policy.any():
                selected_logits = logits[policy, -1].float()
                log_probs = F.log_softmax(selected_logits / (temperature or 1.0), dim=-1)
                sampled = (selected_logits.argmax(dim=-1) if temperature == 0 else
                           torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1))
                token[policy] = sampled
                log_values[policy] = log_probs.gather(-1, sampled[:, None]).squeeze(-1)
                entropy[policy] = -(log_probs.exp() * log_probs).sum(dim=-1)
            for lane in forced.nonzero().flatten().tolist():
                token[lane] = states[lane].pending.popleft()
            tokens.append(token)
            masks.append(valid)
            policy_masks.append(policy)
            old_logs.append(log_values)
            entropies.append(entropy)
            for lane in valid.nonzero().flatten().tolist():
                token_id = int(token[lane])
                if token_id == eos_id:
                    active[lane] = False
                    reasons[lane] = "eos"
                elif tokenizer is not None and tokenizer.protocol == "reasoning_v3":
                    try:
                        states[lane].observe(
                            tokenizer.decode([token_id]), tokenizer, config,
                            position=position, cap=cap,
                        )
                    except CallError as exc:
                        active[lane] = False
                        reasons[lane] = str(exc)
            sequences = torch.cat((sequences, token[:, None]), dim=-1)
            if not active.any():
                break
    finally:
        model.train(was_training)
    def grouped(values):
        return torch.stack(values, dim=-1).view(batch, group_size, -1).detach()
    return Rollouts(
        grouped(tokens), grouped(masks), grouped(old_logs), grouped(entropies),
        grouped(policy_masks), tuple(reasons), tuple(tuple(state.events) for state in states),
    )
