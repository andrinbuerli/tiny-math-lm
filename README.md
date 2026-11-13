# Tiny Math LM

Character-level decoder-only Transformers trained from scratch at **1M, 10M, and 100M parameters**
for a single **RTX 4090 (24 GB)**. Compares supervised fine-tuning (SFT), Group
Relative Policy Optimization (GRPO), and continued SFT on fixed evaluation questions.

Five DeepMind Mathematics Dataset modules: `arithmetic__add_or_sub`,
`arithmetic__mul`, `arithmetic__div`, `arithmetic__mixed`, and `algebra__linear_1d`.
One Unicode character per token, disjoint SFT/RLVR/evaluation pools,
completion-only SFT loss, and deterministic answer verification.

## Results

**Continued SFT leads in 7 of 9 comparisons at matched estimated training FLOPs.**
GRPO leads on number-length OOD at 1M and 10M. Both start from the same SFT
checkpoint for each size.

[![Baseline and reasoning-v3 SFT accuracy at 1M, 10M, and 100M parameters across test and both OOD splits.](docs/assets/reasoning-v3-scaling.png)](docs/assets/reasoning-v3-scaling.svg)

*Baseline vs. reasoning-v3 SFT. OOD means out of distribution.*

[![Accuracy gains over SFT for GRPO and compute-matched continued SFT across model sizes and evaluation splits.](docs/assets/reasoning-v3-training-gains.png)](docs/assets/reasoning-v3-training-gains.svg)

*Additional training gains in percentage points. Continued SFT matches GRPO's
estimated dense FLOPs; [budgets and accounting](docs/compute-matched-continued-sft-plan.md).*

Single-seed results on 50,000 test, 5,000 number-length OOD, and 2,000
expression-length OOD questions. Reasoning-v3 combines scratchpads, subtask
examples, and exact arithmetic assistance on eligible compound tasks; these
results do not isolate their effects. OOD pools use deterministic held-out
questions with difficulty thresholds because the selected modules lacked eligible
official extrapolation rows at the pinned revision.
[Exact results and provenance](docs/results/reasoning-v3-seed18.json) ·
[Protocol](docs/REASONING_V3.md) · [OOD construction](docs/USAGE.md#data-and-ood-protocol-note).

## Quick start

Requires **Python 3.11+**, **uv**, and CUDA-compatible PyTorch. From a checkout
mounted at `/workspace`:

```bash
cd /workspace
uv sync --locked
export WANDB_ENTITY=your-wandb-entity
uv run --locked python scripts/prepare_data.py experiment=02_pilot seed=17
uv run --locked python scripts/train_sft.py \
  experiment=02_pilot model=1m trainer=sft seed=17
```

Experiments require W&B credentials in the environment or ignored `/workspace/.env`
([template](.env.example)). This pilot uses 20,000 answer-only SFT examples and
requires the source dataset download. The results above use the
[reasoning-v3 preset](configs/experiment/15_reasoning_v3_arithmetic_subtasks.yaml),
which requires separately prepared data and checkpoints.

Runs save sanitized resolved Hydra configs, seeds, data/tokenizer hashes,
checkpoint lineage, and optimizer/RNG state for resuming. Generated data,
checkpoints, and full reports are excluded from Git.

Local tests require no W&B connection:

```bash
uv run --locked pytest -q
```

## Guides

- [Usage](docs/USAGE.md): data, training, evaluation, resume, and local apps.
- [Experiments](docs/EXPERIMENT_GUIDE.md): presets and comparison workflows.
- [Reasoning-v3](docs/REASONING_V3.md): scratchpads, arithmetic calls, and rewards.
- [Results](docs/results/reasoning-v3-seed18.json): metrics and provenance.
