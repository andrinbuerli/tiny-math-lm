# TinyMath-RL

TinyMath-RL is a reproducible research codebase for comparing character-level
decoder-only Transformers at approximately 1M, 10M and 100M parameters on basic
mathematics. It compares supervised fine-tuning (SFT), GRPO with verifiable
rewards (RLVR), and a continued-SFT compute control.

The data pipeline is restricted to five DeepMind Mathematics Dataset modules:
addition/subtraction, multiplication, division, mixed arithmetic, and linear
equations. A frozen tokenizer maps each Unicode code point to one ordinary
token. Training and evaluation are configured with Hydra, dependencies are
locked with `uv`, and experimental runs are tracked in Weights & Biases (W&B).

## Quick start

Run commands from `/workspace`. The repository uses the existing CUDA/PyTorch
environment and the local W&B credential file; credentials are read at runtime
and must not be put in command arguments or configuration files.

```bash
uv sync --locked
uv run python scripts/prepare_data.py experiment=01_smoke
uv run python scripts/train_sft.py model=1m trainer=sft experiment=02_pilot
```

Browse prepared examples locally with Streamlit:

```bash
uv run streamlit run scripts/browse_data.py
```

The browser discovers bundles under `data/processed/` and displays the
manifest counts, split examples, and module/category breakdowns. For annotated
reasoning v1 and v2 bundles, it transcribes each reference scratchpad into
plain-language steps and shows the original trace with its notation key.
Prepare a bundle first if the browser reports that none are available.

To run the configured three-size workflow, including SFT, GRPO, continued SFT,
evaluation and comparison reporting:

```bash
uv run python scripts/run_all.py experiment=03_main seed=17
```

Every entry point accepts Hydra overrides. For example, configure W&B metadata
without exposing credentials:

```bash
uv run python scripts/train_sft.py model=10m trainer=sft experiment=02_pilot \
  seed=18 tracking.project=tinymath-rl tracking.group=scale-seed18 \
  tracking.name=pilot-10m-seed18 tracking.mode=online
```

## Latest results: Experiment 13 reasoning-v3

Experiment 13 compares reasoning-v3 against a frozen baseline on seed 18.
Evaluations use the same parent-question identities in both conditions: 50,000
test questions, 5,000 number-length OOD questions, and 2,000 expression-length
OOD questions. The table reports reasoning-v3 accuracy and its percentage-point
change from the matching frozen baseline.

| Model | Split (questions) | SFT accuracy (Δ pp) | Continued-SFT accuracy (Δ pp) |
|---|---|---:|---:|
| 1M | Test (50,000) | 65.76% (+24.29) | 77.17% (+28.63) |
| 1M | Number-length OOD (5,000) | 51.60% (+7.34) | 57.18% (+10.82) |
| 1M | Expression-length OOD (2,000) | 15.25% (+10.95) | 28.70% (+19.40) |
| 10M | Test (50,000) | 86.98% (+38.37) | 86.97% (+36.66) |
| 10M | Number-length OOD (5,000) | 68.34% (+20.20) | 67.48% (+19.44) |
| 10M | Expression-length OOD (2,000) | 47.35% (+34.70) | 47.05% (+30.30) |
| 100M | Test (50,000) | 91.88% (+43.00) | 89.92% (+38.35) |
| 100M | Number-length OOD (5,000) | 76.88% (+27.52) | 71.60% (+21.90) |
| 100M | Expression-length OOD (2,000) | 63.90% (+54.55) | 57.25% (+42.40) |

The reasoning-v3 SFT arm improves over baseline across all three sizes and
splits. The largest gain is 54.55 pp on expression-length OOD for the 100M
model. GRPO was not run for this reasoning-v3 comparison; its rows in the source
report contain baseline measurements only. Results cover one seed, so they do
not estimate between-seed uncertainty. The annotation audit checked 1,096,065
rows and found no invalid traces.

![Experiment 13 reasoning-v3 accuracy against the frozen baseline for test, number-length OOD, and expression-length OOD splits. GRPO reasoning-v3 results were not run.](reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/accuracy_comparison.png)

See the [full comparison report](reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/intermediate_results.md), [CSV results](reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/comparison.csv), and [scalable SVG plot](reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/accuracy_comparison.svg).

## Documentation

- [Usage guide](docs/USAGE.md): environment setup, data preparation, training,
  evaluation, checkpoints, qualitative logging, and W&B synchronization.
- [Experiment guide](docs/EXPERIMENT_GUIDE.md): the research protocol,
  reproducible presets, evaluation workflow and validation commands.
- [Implementation plan](IMPLEMENTATION_PLAN.md): full scientific and
  engineering requirements for this project.

Generated datasets, checkpoints, run outputs, reports, W&B state and secrets
are local artifacts and are excluded from Git.
