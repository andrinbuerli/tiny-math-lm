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
manifest counts, split examples, and module/category breakdowns. Prepare a
bundle first if the browser reports that none are available.

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

## Documentation

- [Usage guide](docs/USAGE.md): environment setup, data preparation, training,
  evaluation, checkpoints, qualitative logging, and W&B synchronization.
- [Experiment guide](docs/EXPERIMENT_GUIDE.md): the research protocol,
  reproducible presets, evaluation workflow and validation commands.
- [Implementation plan](IMPLEMENTATION_PLAN.md): full scientific and
  engineering requirements for this project.

Generated datasets, checkpoints, run outputs, reports, W&B state and secrets
are local artifacts and are excluded from Git.
