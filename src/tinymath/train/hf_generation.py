"""Incremental native-token generation with exact arithmetic observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from tinymath.data.reasoning import task_allows_calls
from tinymath.generation import RolloutConfig
from tinymath.rewards.calls import CallError, execute_call


@dataclass(frozen=True)
class NativeRollout:
    token_ids: list[int]
    policy_mask: list[bool]
    old_log_probs: list[float]
    entropy: list[float]
    termination_reason: str
    call_events: tuple[dict[str, Any], ...]

    @property
    def policy_tokens(self) -> int:
        return sum(self.policy_mask)

    @property
    def executor_tokens(self) -> int:
        return len(self.policy_mask) - self.policy_tokens


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

def _mask_non_token_logits(logits: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    vocab_size = len(tokenizer)
    if logits.shape[-1] < vocab_size:
        raise ValueError("Model output vocabulary is smaller than the pinned native tokenizer")
    if logits.shape[-1] > vocab_size:
        logits = logits.clone()
        logits[..., vocab_size:] = -torch.inf
    return logits


def _active_scratchpad(text: str) -> tuple[str | None, int | None]:
    marker = "<think>\n"
    start = text.find(marker)
    if start < 0:
        return None, None
    scratch_start = start + len(marker)
    end = text.find("</think>", scratch_start)
    if end >= 0:
        return None, None
    return text[scratch_start:], scratch_start


@torch.no_grad()
def generate_native_rollout(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    row: dict[str, Any],
    *,
    context_length: int,
    max_completion_tokens: int,
    temperature: float,
    rollout: RolloutConfig | None = None,
    greedy: bool = False,
) -> NativeRollout:
    """Generate one completion, decoding full accumulated text across token boundaries."""
    config = RolloutConfig.from_config(rollout)
    if config.executor != "exact":
        raise ValueError("Experiment 16 reasoning rollouts require the exact arithmetic executor")
    if not prompt_ids or max_completion_tokens < 1:
        raise ValueError("A nonempty prompt and positive completion cap are required")
    room = min(max_completion_tokens, context_length - len(prompt_ids))
    if room < 1:
        raise ValueError("Prompt leaves no room for a completion")
    if not greedy and temperature <= 0:
        raise ValueError("Sampled GRPO rollouts require positive temperature")

    device = next(model.parameters()).device
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention = torch.ones_like(prompt)
    was_training = model.training
    model.eval()
    output = model(input_ids=prompt, attention_mask=attention, use_cache=True)
    if output.past_key_values is None:
        raise RuntimeError("Qwen model did not return KV cache for incremental generation")
    past = output.past_key_values
    logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
    processed_length = len(prompt_ids)
    generated: list[int] = []
    policy_mask: list[bool] = []
    old_log_probs: list[float] = []
    entropy_values: list[float] = []
    events: list[dict[str, Any]] = []
    dispatched_lines: set[int] = set()
    allowed = task_allows_calls(row)
    call_count = 0
    reason = "budget_exceeded"
    try:
        for _position in range(room):
            if len(generated) >= room:
                reason = "budget_exceeded"
                break
            effective_temperature = temperature if temperature > 0 else 1.0
            log_distribution = F.log_softmax(logits / effective_temperature, dim=-1)
            if greedy:
                token_id = int(logits.argmax(dim=-1).item())
            else:
                token_id = int(torch.multinomial(log_distribution.exp(), num_samples=1).item())
            generated.append(token_id)
            policy_mask.append(True)
            old_log_probs.append(float(log_distribution[0, token_id].item()))
            entropy_values.append(float(-(log_distribution.exp() * log_distribution).sum(-1).item()))
            if token_id == tokenizer.eos_token_id:
                reason = "eos"
                break

            next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            processed_length += 1
            output = model(
                input_ids=next_token,
                attention_mask=torch.ones((1, processed_length), dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)

            text = _decode(tokenizer, generated)
            scratchpad, scratch_start = _active_scratchpad(text)
            if scratchpad is None or scratch_start is None:
                continue
            line_start = scratch_start + scratchpad.rfind("\n") + 1
            line = text[line_start:]
            if not line.lstrip().startswith("call:"):
                # A complete, unhandled call line means the tokenizer crossed a boundary
                # before the executor could observe its exact request suffix.
                completed_lines = scratchpad.split("\n")[:-1]
                cursor = scratch_start
                for completed in completed_lines:
                    if completed.lstrip().startswith("call:") and cursor not in dispatched_lines:
                        raise CallError("malformed_request")
                    cursor += len(completed) + 1
                continue
            if line_start in dispatched_lines:
                continue
            if not allowed:
                raise CallError("disallowed_call")
            if " -> " not in line:
                if "\n" in line:
                    raise CallError("malformed_request")
                continue
            if not line.endswith(" -> "):
                raise CallError("malformed_request")
            if call_count >= config.max_calls:
                raise CallError("call_limit")
            call = execute_call(line, max_scalar_chars=config.max_scalar_chars)
            result_ids = tokenizer.encode(call.result, add_special_tokens=False)
            if not result_ids:
                raise CallError("unencodable_result")
            # Leave one position for the model to continue and emit its terminal EOS.
            if len(generated) + len(result_ids) + 1 > room:
                raise CallError("budget_exceeded")
            call_count += 1
            result_start = len(generated)
            events.append(
                {
                    "operation": call.operation,
                    "left": call.left,
                    "right": call.right,
                    "result": call.result,
                    "request_end": len(generated),
                    "result_start": result_start,
                    "result_end": result_start + len(result_ids),
                }
            )
            dispatched_lines.add(line_start)
            generated.extend(int(value) for value in result_ids)
            policy_mask.extend([False] * len(result_ids))
            old_log_probs.extend([0.0] * len(result_ids))
            entropy_values.extend([0.0] * len(result_ids))
            observation = torch.tensor([result_ids], dtype=torch.long, device=device)
            processed_length += len(result_ids)
            output = model(
                input_ids=observation,
                attention_mask=torch.ones((1, processed_length), dtype=torch.long, device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
    except CallError as exc:
        reason = str(exc)
    finally:
        model.train(was_training)
    return NativeRollout(
        token_ids=generated,
        policy_mask=policy_mask,
        old_log_probs=old_log_probs,
        entropy=entropy_values,
        termination_reason=reason,
        call_events=tuple(events),
    )
