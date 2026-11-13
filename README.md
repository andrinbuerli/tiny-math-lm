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

Reasoning-v3 final-answer accuracy, **seed 18**. Bold marks the best arm per row.
OOD means out of distribution.

| Model | Split | SFT | GRPO | Continued SFT¹ |
|:---|:---|---:|---:|---:|
| 1M | Test | 65.76% | 69.02% | **69.49%** |
| 1M | Number-length OOD | 51.60% | **57.08%** | 53.20% |
| 1M | Expression-length OOD | 15.25% | 16.30% | **19.70%** |
| 10M | Test | 86.98% | 87.20% | **88.41%** |
| 10M | Number-length OOD | 68.34% | **71.28%** | 70.24% |
| 10M | Expression-length OOD | 47.35% | 47.90% | **49.80%** |
| 100M | Test | 91.88% | 91.10% | **92.98%** |
| 100M | Number-length OOD | 76.88% | 76.72% | **78.92%** |
| 100M | Expression-length OOD | 63.90% | 60.55% | **68.30%** |

¹ Additional training compute matches GRPO using estimated dense FLOPs, not
hardware measurements. [Budgets and accounting](docs/compute-matched-continued-sft-plan.md).

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
- [Results](docs/results/reasoning-v3-seed18.json): metrics and provenance;
  [scaling figure](docs/assets/reasoning-v3-scaling.svg) and
  [training gains](docs/assets/reasoning-v3-training-gains.svg).
