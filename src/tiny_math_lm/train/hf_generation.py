"""Incremental native-token generation with exact arithmetic observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from tiny_math_lm.checkpoint import capture_rng_state, restore_rng_state
from tiny_math_lm.data.reasoning import task_allows_calls
from tiny_math_lm.generation import RolloutConfig
from tiny_math_lm.rewards.calls import CallError, execute_call


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


def _select_cache_rows(past: Any, indices: torch.Tensor) -> Any:
    select = getattr(past, "batch_select_indices", None)
    if callable(select):
        select(indices)
        return past
    if isinstance(past, tuple):
        return tuple(
            tuple(
                value.index_select(0, indices) if isinstance(value, torch.Tensor) else value
                for value in layer
            )
            for layer in past
        )
    raise TypeError("Batched native rollouts require a cache that supports batch row selection")


def _has_executor_request(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    text = _decode(tokenizer, token_ids)
    think_start = text.find("<think>\n")
    if think_start < 0:
        return False
    scratch_start = think_start + len("<think>\n")
    think_end = text.find("</think>", scratch_start)
    scratchpad = text[scratch_start:] if think_end < 0 else text[scratch_start:think_end]
    return any(line.lstrip().startswith("call:") for line in scratchpad.splitlines())


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
            probabilities = log_distribution.exp()
            finite_log_distribution = log_distribution.masked_fill(
                ~torch.isfinite(log_distribution), 0.0
            )
            entropy_values.append(float(-(probabilities * finite_log_distribution).sum(-1).item()))
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


@torch.no_grad()
def generate_native_rollout_group(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    row: dict[str, Any],
    *,
    group_size: int,
    context_length: int,
    max_completion_tokens: int,
    temperature: float,
    rollout: RolloutConfig | None = None,
    greedy: bool = False,
) -> tuple[NativeRollout, ...]:
    """Sample one prompt's completion group together when no executor is needed.

    If a sampled trajectory requests execution, discard the speculative batch,
    restore its RNG state, and reproduce the group through the exact incremental
    executor path. This keeps executor observations inside each trajectory.
    """
    if group_size < 1:
        raise ValueError("A rollout group must contain at least one completion")
    if group_size == 1 or not callable(getattr(model, "generate", None)):
        return tuple(
            generate_native_rollout(
                model,
                tokenizer,
                prompt_ids,
                row,
                context_length=context_length,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                rollout=rollout,
                greedy=greedy,
            )
            for _ in range(group_size)
        )
    if not prompt_ids:
        raise ValueError("A nonempty prompt is required")
    room = min(max_completion_tokens, context_length - len(prompt_ids))
    if room < 1:
        raise ValueError("Prompt leaves no room for a completion")
    if not greedy and temperature <= 0:
        raise ValueError("Sampled GRPO rollouts require positive temperature")

    rng_state = capture_rng_state()
    device = next(model.parameters()).device
    prompt = torch.tensor([list(prompt_ids)] * group_size, dtype=torch.long, device=device)
    attention = torch.ones_like(prompt)
    was_training = model.training
    model.eval()
    try:
        generation_options: dict[str, Any] = {
            "input_ids": prompt,
            "attention_mask": attention,
            "max_new_tokens": room,
            "do_sample": not greedy,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": (
                tokenizer.eos_token_id
                if getattr(tokenizer, "pad_token_id", None) is None
                else tokenizer.pad_token_id
            ),
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
        }
        if not greedy:
            generation_options.update(
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                typical_p=1.0,
                repetition_penalty=1.0,
            )
        generated = model.generate(**generation_options)
        sequences = generated.sequences
        scores = generated.scores
        if sequences.shape[0] != group_size or not scores:
            raise RuntimeError("Batched generation returned incomplete rollout sequences or scores")
        token_lists: list[list[int]] = []
        termination: list[str] = []
        for sequence in sequences:
            tokens = [int(value) for value in sequence[len(prompt_ids) :].tolist()]
            try:
                eos_index = tokens.index(tokenizer.eos_token_id)
            except ValueError:
                eos_index = None
            if eos_index is not None:
                tokens = tokens[: eos_index + 1]
                termination.append("eos")
            else:
                tokens = tokens[:room]
                termination.append("budget_exceeded")
            token_lists.append(tokens)

        # The exact executor parser owns any transcript that starts a call.
        for tokens in token_lists:
            text = _decode(tokenizer, tokens)
            think_start = text.find("<think>\n")
            if think_start >= 0:
                scratchpad_start = think_start + len("<think>\n")
                think_end = text.find("</think>", scratchpad_start)
                scratchpad = (
                    text[scratchpad_start:] if think_end < 0 else text[scratchpad_start:think_end]
                )
            else:
                scratchpad = ""
            if any(line.lstrip().startswith("call:") for line in scratchpad.splitlines()):
                restore_rng_state(rng_state)
                return tuple(
                    generate_native_rollout(
                        model,
                        tokenizer,
                        prompt_ids,
                        row,
                        context_length=context_length,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                        rollout=rollout,
                        greedy=greedy,
                    )
                    for _ in range(group_size)
                )

        result: list[NativeRollout] = []
        for row_index, tokens in enumerate(token_lists):
            token_log_probs: list[float] = []
            entropies: list[float] = []
            if len(tokens) > len(scores):
                raise RuntimeError("Batched generation omitted sampled-token scores")
            for token_index, token_id in enumerate(tokens):
                logits = _mask_non_token_logits(scores[token_index][row_index].float(), tokenizer)
                effective_temperature = temperature if temperature > 0 else 1.0
                if greedy:
                    logits = logits / effective_temperature
                log_distribution = F.log_softmax(logits, dim=-1)
                token_log_probs.append(float(log_distribution[token_id].item()))
                probabilities = log_distribution.exp()
                finite = log_distribution.masked_fill(~torch.isfinite(log_distribution), 0.0)
                entropies.append(float(-(probabilities * finite).sum().item()))
            result.append(
                NativeRollout(
                    token_ids=tokens,
                    policy_mask=[True] * len(tokens),
                    old_log_probs=token_log_probs,
                    entropy=entropies,
                    termination_reason=termination[row_index],
                    call_events=(),
                )
            )
        return tuple(result)
    finally:
        model.train(was_training)


@torch.no_grad()
def generate_native_rollout_batch(
    model: Any,
    tokenizer: Any,
    items: Sequence[tuple[Sequence[int], dict[str, Any]]],
    *,
    context_length: int,
    max_completion_tokens: int,
    temperature: float,
    rollout: RolloutConfig | None = None,
    greedy: bool = False,
) -> tuple[NativeRollout, ...]:
    """Incrementally sample a bounded batch while retaining only action statistics.

    A batch containing any executor request is discarded and replayed with the
    exact per-trajectory executor path from the saved RNG state.
    """
    if not items or any(not prompt_ids for prompt_ids, _row in items):
        raise ValueError("A nonempty batch of nonempty prompts is required")
    if max_completion_tokens < 1:
        raise ValueError("A positive completion cap is required")
    if not greedy and temperature <= 0:
        raise ValueError("Sampled GRPO rollouts require positive temperature")
    config = RolloutConfig.from_config(rollout)
    if config.executor != "exact":
        raise ValueError("Experiment 16 reasoning rollouts require the exact arithmetic executor")
    prompts = [list(prompt_ids) for prompt_ids, _row in items]
    rooms = [min(max_completion_tokens, context_length - len(prompt)) for prompt in prompts]
    if any(room < 1 for room in rooms):
        raise ValueError("A prompt leaves no room for a completion")

    rng_state = capture_rng_state()
    device = next(model.parameters()).device
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = tokenizer.eos_token_id if pad_id is None else pad_id
    width = max(map(len, prompts))
    input_ids = torch.full((len(items), width), pad_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for index, prompt_ids in enumerate(prompts):
        input_ids[index, width - len(prompt_ids) :] = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        )
        attention[index, width - len(prompt_ids) :] = 1
    position_ids = attention.cumsum(dim=-1).sub(1).clamp_min(0)
    was_training = model.training
    model.eval()
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
            use_cache=True,
        )
        past = output.past_key_values
        if past is None:
            raise RuntimeError("Qwen model did not return KV cache for incremental generation")
        logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)
        active_indices = list(range(len(items)))
        generated: list[list[int]] = [[] for _ in items]
        old_log_probs: list[list[torch.Tensor]] = [[] for _ in items]
        entropies: list[list[torch.Tensor]] = [[] for _ in items]
        reasons: list[str | None] = [None] * len(items)
        for offset in range(max(rooms, default=0)):
            effective_temperature = temperature if temperature > 0 else 1.0
            log_distribution = F.log_softmax(logits / effective_temperature, dim=-1)
            if greedy:
                sampled = logits.argmax(dim=-1)
            else:
                sampled = torch.multinomial(log_distribution.exp(), num_samples=1).squeeze(-1)
            token_ids = sampled.tolist()
            log_probs = log_distribution.gather(-1, sampled[:, None]).squeeze(-1)
            probabilities = log_distribution.exp()
            finite = log_distribution.masked_fill(~torch.isfinite(log_distribution), 0.0)
            entropy = -(probabilities * finite).sum(dim=-1)
            keep_local: list[int] = []
            next_tokens: list[int] = []
            for local_index, token_id in enumerate(token_ids):
                original_index = active_indices[local_index]
                generated[original_index].append(int(token_id))
                old_log_probs[original_index].append(log_probs[local_index])
                entropies[original_index].append(entropy[local_index])
                if token_id == tokenizer.eos_token_id:
                    reasons[original_index] = "eos"
                elif offset + 1 >= rooms[original_index]:
                    reasons[original_index] = "budget_exceeded"
                else:
                    keep_local.append(local_index)
                    next_tokens.append(int(token_id))
            if not keep_local:
                break
            if len(keep_local) < len(active_indices):
                cache_indices = torch.tensor(keep_local, dtype=torch.long, device=device)
                past = _select_cache_rows(past, cache_indices)
                attention = attention.index_select(0, cache_indices)
                active_indices = [active_indices[index] for index in keep_local]
            token = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
            attention = torch.cat(
                [
                    attention,
                    torch.ones((len(next_tokens), 1), dtype=attention.dtype, device=device),
                ],
                dim=1,
            )
            position_ids = attention.cumsum(dim=-1)[:, -1:].sub(1).clamp_min(0)
            output = model(
                input_ids=token,
                attention_mask=attention,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            logits = _mask_non_token_logits(output.logits[:, -1, :].float(), tokenizer)

        if any(_has_executor_request(tokenizer, tokens) for tokens in generated):
            restore_rng_state(rng_state)
            return tuple(
                generate_native_rollout(
                    model,
                    tokenizer,
                    prompt_ids,
                    row,
                    context_length=context_length,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    rollout=config,
                    greedy=greedy,
                )
                for prompt_ids, row in items
            )

        result = []
        for index, tokens in enumerate(generated):
            if reasons[index] is None:
                raise RuntimeError(
                    "Batched rollout ended without EOS or completion-budget termination"
                )
            token_log_probs = torch.stack(old_log_probs[index]).detach().cpu().tolist()
            token_entropies = torch.stack(entropies[index]).detach().cpu().tolist()
            result.append(
                NativeRollout(
                    token_ids=tokens,
                    policy_mask=[True] * len(tokens),
                    old_log_probs=[float(value) for value in token_log_probs],
                    entropy=[float(value) for value in token_entropies],
                    termination_reason=reasons[index],
                    call_events=(),
                )
            )
        return tuple(result)
    finally:
        model.train(was_training)
