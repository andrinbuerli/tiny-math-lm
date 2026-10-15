# Reasoning v3 execution and rewards

V3 supports ordinary character-tokenized scratchpads and optional arithmetic calls.
Calls are optional: a supported forward derivation with a correct final answer can
earn full reward without calling the executor. The scorer checks trace grammar,
every call's exact arithmetic, and the final answer. It does not prove that all
intermediate derivations solve the original question or require reference equality.

Tool access follows `proper_subexpression_v1`, shared by reference annotation,
generation, GRPO rewards, evaluation, and qualitative previews. Atomic arithmetic
and standalone subtasks cannot call tools. Compound arithmetic can call only strict
source subexpressions and must solve its final operation locally. Requests equivalent
to the root operation or unrelated to the question are forbidden. Algebra may call
the arithmetic division derived after collecting and moving equation terms.

For `Calculate (25+32)*2.`, `call:add(25,32)` is permitted; `call:multiply(57,2)`
is not. `Calculate 25+32.` must use a local worked derivation without any calls.
Rational literals are atomic operands, including in a single-operation parent task.
Alternative local derivations remain allowed; tool requests use the conservative
source-subexpression allowlist. Eligibility is derived without reading the target answer.

The reward is `0.1 * format_valid + 0.9 * full_success`. Formatting requires exactly
one nonempty `<scp>` block, one terminal `A: <scalar>` line, and EOS. Full success
also requires a valid trace and a correct answer. A well-formatted wrong answer or
invalid derivation earns 0.1. Malformed, capped, or executor-failed rollouts earn 0.
Forbidden delegation earns 0 even if formatting and the final answer are correct.
This v3-specific partial reward intentionally replaces the original binary reward;
answer-only, v1, and v2 keep their existing objectives.

With `rollout.executor=exact`, generation pauses after a line-start request inside
the scratchpad, such as `call:add(25,32) -> `. The local rational executor inserts
`57`; the policy generates the following newline and continues. Supported operations
are `add`, `subtract`, `multiply`, and `divide`. Operands/results use canonical
integers or reduced fractions. No model workers, target answers, or generated code
are used by the executor. Malformed completed requests, division by zero, excessive
calls/scalars, and unencodable results terminate that lane with zero reward.
Forbidden requests terminate before arithmetic execution or result insertion, including
with `rollout.executor=none`. Explicit task metadata must match the prompt; standalone
subtask status is preserved. Unknown questions fail closed for tool access.

The transcript mask includes inserted results and EOS. The policy mask excludes
inserted results and padding from the clipped loss, reference KL, entropy, clipping
statistics, and normalization. Later policy tokens condition on the entire transcript.
The optional KL compares current and frozen-reference probabilities at policy actions.
Group-relative advantages use the shaped scalar rewards; constant groups, including
all-0.1 groups, have zero advantage and skip backward/optimizer updates. Group metrics
distinguish full-success, zero, format-only, and mixed-reward groups.

The configured completion ceiling and remaining model context bound the total
transcript, including inserted characters and EOS. Results are checked before
insertion. No context is truncated. Default executor limits are 32 calls and 128
characters per operand/result. Existing configurations disable execution by default;
the `reasoning_v3` preset enables exact execution with a 512-token ceiling.

```bash
# In a checkout mounted elsewhere, export WORKSPACE_ROOT to its absolute path first.
uv run python scripts/train_rlvr.py --config-name reasoning_v3 \
  init_checkpoint=runs/<v3-sft-run>/best.pt
uv run python scripts/evaluate.py --config-name reasoning_v3 \
  checkpoint=runs/<v3-grpo-run>/latest_inference.pt

# Use a distinct run identity for an unassisted comparison.
uv run python scripts/train_rlvr.py --config-name reasoning_v3 \
  rollout.executor=none experiment.name=v3_unassisted \
  init_checkpoint=runs/<v3-sft-run>/best.pt
```

RLVR draws parent prompts only from `rlvr.jsonl`. Main evaluation and qualitative
tables also select parent rows; generated subtask rows do not affect parent accuracy.
Evaluation reports final-answer accuracy separately from full-success rate and mean
reward. Assisted mode, reward weights, limits, implementation versions, and token
accounting are persisted in configs, checkpoint metadata, evaluation identity, and
W&B. Incompatible checkpoint/evaluation resumes are rejected. Policy, executor, and
total transcript token counts remain separate; compute estimates include inserted
tokens. Reference-trace equality is diagnostic only.

Generated subtask schema v2 treats `call_operands` as two atomic rational scalars.
The question still displays fractions, but their slashes are not parsed as extra
division tasks. `subtask_operation_count` is always 1, and `module` follows
`call_operation`, including for fractional operands. The annotator validates the
question, operation, operands, result, and module together before producing the
ordinary v2 worked trace. Such subtasks cannot recursively delegate their operation.

Regenerate with `uv run python scripts/annotate_reasoning.py --config-name reasoning_v3`.
The proper-subexpression annotation has its own resumable W&B identity and replaces the standard
v3 bundle only after successful validation. A hidden sibling backup preserves the
previous bundle; hidden backups and staging directories are excluded from the browser.
The new annotation hash intentionally invalidates compatibility with checkpoints
trained on the old annotations.

## Controlled validation

```bash
# Run twice with identical overrides: checkpoint interruption, then same-run resume.
uv run python scripts/validate_reasoning_v3.py
uv run python scripts/validate_reasoning_v3.py
```

This creates disjoint synthetic parent fixtures across all five modules under
`runs/v3-proper-subexpression-validation`. It uses a tiny randomly initialized GPT and scripted
training trajectories to exercise exact call execution, mixed 0/0.1/1 rewards,
optimizer updates, and checkpoint/W&B resume. These off-policy controlled trajectories
include forbidden requests that must be denied before execution. Every update covers
all five modules, including direct atomic arithmetic and permitted compound delegation.
The scripted trajectories and entropy are plumbing checks, not learning measurements. The final
greedy evaluation uses real model outputs. Unit tests also compare interrupted and
uninterrupted optimizer results. Run `uv run pytest -q` for the regression suite.

All experiment invocations require W&B. Credentials load at runtime from the workspace
`.env` or environment without entering configs. If online tracking is unavailable,
explicitly select `tracking.mode=offline`; those runs remain unsynced, and offline
resume does not demonstrate server-side W&B resume. To sync queued data later, run
`uv run wandb sync <offline-run-dir>`. Generated datasets, checkpoints, and run data
stay local and untracked.
