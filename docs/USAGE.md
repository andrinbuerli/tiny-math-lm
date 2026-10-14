# TinyMath-RL usage

The repository uses `uv`, Hydra and W&B for experiment setup and tracking. Run commands from `/workspace`.

## Environment and prepared data

```bash
cd /workspace
export PATH="/workspace/.tools:$PATH"
uv sync --locked --extra dev
uv run python scripts/prepare_data.py experiment=01_smoke
```

The default W&B mode is `online`. The tracking helper reads `WANDB_API_KEY` from the environment or `/workspace/.env` at runtime. It also accepts the legacy `WANDBKEY` assignment found in the supplied workspace file. To deliberately create a local, unsynced run when authentication or network access is unavailable, pass `tracking.mode=offline`; every such run records `tracking/sync_status=unsynced`. Its local W&B directory is printed at the end and can later be uploaded with:

```bash
uv run wandb sync --yes --entity <wandb-entity> --project <wandb-project> \
  /workspace/runs/<experiment>/<model>/<stage>/seed<seed>/wandb/offline-run-<timestamp>-<id>
```

The W&B CLI does not load `/workspace/.env` itself. Before running this command, load its key into the `WANDB_API_KEY` environment variable with `python-dotenv` (the experiment tracking helper does this automatically), without printing the value or passing it as a command-line argument. Use the entity from the authenticated W&B account and the project recorded in the run's `wandb_run.json`. Add `--include-synced` when retrying a run that the CLI has already marked as synced locally.

The data config pins `d0rj/mathematics_dataset` to revision `d9f78d0665b3a13dad21c6830b9b358c8a25ae74`, matching the prepared bundle used by the current 100k runs. To reproduce against a different revision, set `data.revision=<commit-sha>` explicitly.

The saved `wandb_run.json` records the run ID and whether synchronization is pending, synced or unsynced. Never pass the API key as a command-line override. The installed W&B SDK warns that `resume` is ignored in offline mode: training checkpoints and RNG state resume locally, but W&B starts another offline history directory with the same configured run ID. Online mode uses W&B's normal resume behavior.

## Train and evaluate

```bash
uv run python scripts/train_sft.py model=1m trainer=sft experiment=02_pilot
uv run python scripts/train_rlvr.py model=1m rlvr=grpo experiment=02_pilot \
  'init_checkpoint=runs/pilot/1m/sft/seed17/best.pt'
uv run python scripts/evaluate.py model=1m eval=default experiment=02_pilot \
  'checkpoint=runs/pilot/1m/grpo/seed17/latest.pt' eval.greedy=true \
  eval.max_new_tokens=128
uv run python scripts/train_sft.py model=1m trainer=continued_sft experiment=02_pilot \
  'init_checkpoint=runs/pilot/1m/sft/seed17/best.pt'
```

Training and evaluation locate the prepared bundle by experiment and seed. Set `data_bundle=data/processed/<bundle-name>` to select one explicitly. Each stage writes its sanitized composed config, checkpoints, CSV metrics and W&B run state beneath `/workspace/runs`.

Checkpoint evaluation is greedy by default (`eval.greedy=true`, temperature 0). `eval.max_new_tokens` sets the generation cap; remaining context can lower the effective cap for long prompts. `results.json` and comparison reports record the strategy, temperature, cap and model context length. For an explicitly sampled evaluation, set `eval.greedy=false eval.temperature=0.8`.

SFT, continued-SFT and GRPO training runs log deterministic W&B qualitative tables every 5,000 optimizer steps and at the final step, with generated answers for up to two examples per module from the training, validation and test pools. At those same points, they log greedy exact accuracy for the same fixed examples under `train/greedy_accuracy`, `val/greedy_accuracy`, and `test/greedy_accuracy`; with five modules and two examples per module, each value summarizes 10 examples and is not full-split accuracy. To attach a final table to a run that began before periodic qualitative logging was added, rerun the Hydra helper with that run's original config overrides; it resumes the matching W&B run ID:

```bash
uv run python scripts/log_qualitative.py model=1m experiment=03_main seed=18 \
  tracking.mode=online data_bundle=data/processed/<bundle-name>
```

Use `trainer=continued_sft` for the continued-SFT run. For GRPO, use `trainer=sft` plus the corresponding `init_checkpoint=...` override used to launch that GRPO run.

To compare examples over time, open a run's W&B History and select `qualitative/examples`; use `optimizer_step` as the x-axis and open the table at each 5,000-step point. The selected question, target, split and module rows are fixed across those tables, while `prediction`, `correct` and `format_valid` show how the model changes.

`run_all.py experiment=02_pilot` prepares data once and runs SFT, GRPO, continued-SFT and their evaluations for the configured model list. Override `orchestration.models=[1m]` to run one size. The default `01_smoke` preset is data-only for orchestration; direct stage commands can still run short optimization checks by setting `trainer.max_steps` or `rlvr.updates`.

Model pipelines run serially by default. To allow overlap, set `orchestration.parallel_models=true`; before each training or evaluation process, the orchestrator checks GPU utilization and free VRAM with `nvidia-smi`. It waits until utilization is at most 75% and free VRAM covers the configured model reserve plus 2 GiB, then waits 15 seconds after launch for telemetry to reflect the new process. Thresholds and polling interval are in `configs/runtime/default.yaml`. If GPU telemetry cannot be read, the orchestrator fails instead of starting an unchecked optimization process.

The 100k-step restart presets use the same prepared bundle and can start from fresh random initialization:

```bash
uv run python scripts/run_all.py experiment=08_restarted_100k_greedy \
  data_bundle=data/processed/<bundle-name> seed=18
```

They save resumable training checkpoints and `latest_inference.pt` under each run directory at the configured checkpoint interval, currently every 500 steps.

## W&B overrides and local report

```bash
uv run python scripts/train_sft.py experiment=02_pilot model=1m \
  tracking.project=tinymath-rl tracking.group=pilot-1m \
  tracking.name=pilot-1m-seed17 tracking.mode=online
uv run python scripts/plot.py experiment=02_pilot tracking.mode=online
```

The comparison report writes `report.md`, `comparison.csv`, `comparison.json`, and PNG/SVG figures under `/workspace/reports/<experiment>/seed<seed>`. It lists missing model/arm/split measurements as missing and includes only measurements from local evaluation files. The Markdown report summarizes ID and OOD accuracy, gains over SFT, GRPO group diagnostics, and descriptive token/FLOP efficiency. Its limitations distinguish SFT processed input positions from GRPO generated completion tokens. The comparison W&B run logs the figures and publishes these report files as an artifact. SFT compute uses `6PN`; GRPO includes rollout prefix recomputation and policy/reference token positions under dense `2N/6N` multipliers. These are approximations, not hardware-counter FLOP measurements.

## Data and OOD protocol note

The streamed English split contains the official `interpolate` evaluation category for all five selected modules. The exact selected module IDs had no eligible official `extrapolate` rows in the inspected dataset revision. When a module lacks those rows, preparation reserves a deterministic capped set from its eligible `train-*` rows using the interpolation p95 thresholds and removes those questions from SFT, validation and RLVR. Number-width and expression-length groups each change one measured feature. The manifest records the source category and thresholds per module; a feature with no eligible candidates remains empty.
Each saved training directory contains `latest.pt` for resuming and
`latest_inference.pt` with model weights and tokenizer/model metadata for later
inference. Both are refreshed at the configured checkpoint interval and at the
final step.
