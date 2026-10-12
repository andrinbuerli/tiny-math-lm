# TinyMath-RL usage

The repository uses `uv`, Hydra and W&B for experiment setup and tracking. Run commands from `/workspace`.

## Environment and prepared data

```bash
cd /workspace
uv sync --locked --extra dev
uv run python scripts/prepare_data.py experiment=smoke
```

The default W&B mode is `online`. The tracking helper reads `WANDB_API_KEY` from the environment or `/workspace/.env` at runtime. To deliberately create a local, unsynced run when authentication or network access is unavailable, pass `tracking.mode=offline`; every such run records `tracking/sync_status=unsynced`. Its local W&B directory is printed at the end and can later be uploaded with:

```bash
uv run wandb sync /workspace/runs/<experiment>/<model>/<stage>/seed<seed>/wandb/offline-run-<timestamp>-<id>
```

The saved `wandb_run.json` records the run ID and whether synchronization is pending, synced or unsynced. Never pass the API key as a command-line override.

## Train and evaluate

```bash
uv run python scripts/train_sft.py model=1m trainer=sft experiment=pilot
uv run python scripts/train_rlvr.py model=1m rlvr=grpo experiment=pilot \
  'init_checkpoint=runs/pilot/1m/sft/seed17/best.pt'
uv run python scripts/evaluate.py model=1m eval=default experiment=pilot \
  'checkpoint=runs/pilot/1m/grpo/seed17/latest.pt'
uv run python scripts/train_sft.py model=1m trainer=continued_sft experiment=pilot \
  'init_checkpoint=runs/pilot/1m/sft/seed17/best.pt'
```

Training and evaluation locate the prepared bundle by experiment and seed. Set `data_bundle=data/processed/<bundle-name>` to select one explicitly. Each stage writes its sanitized composed config, checkpoints, CSV metrics and W&B run state beneath `/workspace/runs`.

`run_all.py experiment=pilot` prepares data once and runs SFT, GRPO, continued-SFT and their evaluations for the configured model list. Override `orchestration.models=[1m]` to run one size. The default smoke preset is data-only for orchestration; direct stage commands can still run short optimization checks by setting `trainer.max_steps` or `rlvr.updates`.

## W&B overrides and local report

```bash
uv run python scripts/train_sft.py experiment=pilot model=1m \
  tracking.project=tinymath-rl tracking.group=pilot-1m \
  tracking.name=pilot-1m-seed17 tracking.mode=online
uv run python scripts/plot.py experiment=pilot tracking.mode=online
```

The comparison report writes `comparison.csv`, `comparison.json`, and PNG/SVG figures under `/workspace/reports/<experiment>/seed<seed>`. It lists missing model/arm/split measurements as missing and includes only measurements from local evaluation files. Estimated compute uses `6 × parameters × processed tokens`; this is an approximation, not a measured FLOP count.

## Data and OOD protocol note

The streamed English split contains the official `interpolate` evaluation category for all five selected modules. The exact selected module IDs had no eligible official `extrapolate` rows in the inspected dataset revision. When a module lacks those rows, preparation reserves a deterministic capped set from its eligible `train-*` rows using the interpolation p95 thresholds and removes those questions from SFT, validation and RLVR. Number-width and expression-length groups each change one measured feature. The manifest records the source category and thresholds per module; a feature with no eligible candidates remains empty.
