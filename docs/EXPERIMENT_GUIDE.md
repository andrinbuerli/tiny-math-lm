# TinyMath-RL experiment guide

W&B is for experiments only: training, named optimization pilots/smokes,
checkpoint evaluations, OOD benchmarks and experimental comparisons. Data preparation
and annotation may be tracked as stages of a named experiment. Documentation, code
maintenance, unit/integration tests, configuration checks, local browsing, Git
operations and documentation figures must not create, resume or upload W&B runs.
Routine development requires neither W&B authentication nor W&B status reporting.

## Environment and tracking

Run commands from `/workspace`. Use the checked-in lockfile and the existing CUDA/PyTorch environment:

```bash
uv sync --locked
```

Experimental entry points compose Hydra YAML configurations. W&B tracking is online by default and reads credentials from the existing `/workspace/.env` at runtime. Do not put credentials in commands or configuration files. To set tracking metadata without changing YAML:

```bash
uv run python scripts/train_sft.py model=1m trainer=sft experiment=02_pilot seed=17 tracking.project=tinymath-rl tracking.group=pilot-seed17 tracking.name=pilot-1m-seed17 tracking.mode=online
```

Use `tracking.mode=offline` only when online tracking cannot connect; offline runs are marked unsynced. To upload one later, load its W&B API key into `WANDB_API_KEY` at runtime with `python-dotenv` (the W&B CLI does not read `/workspace/.env`), then run:

```bash
uv run wandb sync --yes --entity <wandb-entity> --project <wandb-project> <offline-run-directory>
```

Use the authenticated account's entity and the project recorded in the run's `wandb_run.json`. Do not print the key or pass it as a command-line argument. Add `--include-synced` when retrying a run that the CLI has already marked synced locally.

## Prepare data and run experiments

The prepared bundle pins the dataset revision and seed, saves the tokenizer and split manifest, and is reused across all model sizes and arms.

```bash
uv run python scripts/prepare_data.py experiment=02_pilot seed=17
```

For the complete three-size SFT, GRPO, continued-SFT, evaluation, and comparison workflow, use the orchestrator:

```bash
uv run python scripts/run_all.py experiment=03_main seed=17
```

The default orchestrator is serial. Set `orchestration.parallel_models=true` only when concurrent model pipelines are useful; each training/evaluation launch checks `nvidia-smi` and waits for utilization and free-VRAM thresholds configured in `configs/runtime/default.yaml`. A missing GPU telemetry command is an error for optimization launches.

The main preset uses 500k unique SFT training examples and 50k SFT steps. The 100k-step scratch restart preset also logs fixed train, validation, and test qualitative examples with greedy accuracy every 5k steps:

```bash
uv run python scripts/run_all.py experiment=10_restarted_100k_greedy seed=18 data_bundle=data/processed/d9f78d0665b3-main-seed18 tracking.mode=online
```

Run a single SFT arm with the same composed configuration system:

```bash
uv run python scripts/train_sft.py model=1m trainer=sft experiment=02_pilot seed=17
```

To start GRPO from the matching SFT checkpoint and evaluate a saved checkpoint, provide the local paths explicitly:

```bash
uv run python scripts/train_rlvr.py model=1m trainer=sft experiment=02_pilot seed=17 'init_checkpoint=runs/pilot/1m/sft/seed17/best.pt'
uv run python scripts/evaluate.py model=1m trainer=sft experiment=02_pilot seed=17 'checkpoint=runs/pilot/1m/grpo/seed17/latest_inference.pt'
uv run python scripts/plot.py experiment=02_pilot seed=17
```

All experiments use the same five selected DeepMind Mathematics modules and a strict single-Unicode-code-point tokenizer. Each evaluation records exact accuracy, format validity, module metrics, and controlled number-length and expression-length OOD results. Training checkpoints and local metrics live under `runs/`; reports and figures are written under `reports/`.

## Validation

Run the unit and integration suite with:

```bash
uv run pytest -q
```

The integration test uses CUDA when available. To force its CPU fallback, run `CUDA_VISIBLE_DEVICES='' uv run pytest -q`. Unit/integration tests must use local fixtures or mocked tracking and must not connect to W&B or create offline runs, even when they exercise optimizer steps.

Every command accepts Hydra overrides. For example, change a model, seed, W&B project, group, or run name by adding `model=10m seed=18 tracking.project=tinymath-rl tracking.group=scale-seed18 tracking.name=pilot-10m-seed18`.
