# TinyMath-RL agent instructions

Follow `IMPLEMENTATION_PLAN.md` in full. Scientific scope: exactly five specified DeepMind Mathematics Dataset modules, strict one-Unicode-character-per-token tokenizer, leakage-free SFT/RLVR/test pools, 1M/10M/100M decoder-only models, completion-only SFT objective, verifiable GRPO and RTX 4090 constraints.

## Mandatory tools and tracking

- **uv only** for Python environment, dependency locking (`pyproject.toml` + tracked `uv.lock`), and `uv run` command execution; do not switch to pip/Poetry/Conda workflows.
- **Hydra/OmegaConf** for composed YAML configurations and CLI overrides for **all** experiment entry points; persist the full sanitized resolved configuration with each run.
- **W&B is for experiments only**: SFT, GRPO, continued-SFT, named optimization pilots/smokes, checkpoint evaluations, OOD benchmarks and experimental comparisons. Log experiment metrics, provenance and results; resume the same W&B run after interruption. Data preparation and annotation may be tracked only as stages of a named experiment.
- Documentation/README work, code edits, refactoring, unit/integration tests, linting, configuration checks, local browsing, Git operations and preparing documentation figures must not create, resume or upload W&B runs. Tests must use local fixtures or mocked tracking, even when they exercise optimization. Do not authenticate to W&B or inspect remote run status for routine development work.
- W&B API key is in the existing **`/workspace/.env`**. Load it only when an experiment requires tracking (e.g., `python-dotenv`), without printing, copying, tracking, uploading or embedding it in configs. Gitignore `.env` and local W&B state. Report missing auth/network and flag offline runs as unsynced only when relevant to an experiment being run or diagnosed.

## Required development behavior

- Work exclusively in `/workspace`; do not modify Docker host configuration.
- Make a **local Git commit after each cohesive, validated implementation unit**. Never push, fetch, pull, configure a remote, or contact a Git remote.
- Inspect current Git status first; preserve unrelated staged/unstaged user changes.
- Before each commit, run relevant tests, inspect diffs and `git diff --check`, stage explicit intended paths only, and use a Conventional Commit message.
- Do not commit datasets, checkpoints, weights, generated run data, credentials, `.env`, `wandb/` or secrets. **Do commit `uv.lock`, Hydra configs, source and tests.**
- At milestones run integration checks when possible. Finish with actual tests, milestones, limitations, local `git log --oneline` and working-tree status. Include W&B tracking/sync status only when experiments were run or diagnosed, or the user explicitly requests it; omit it for routine development and documentation work. Never invent results or commit hashes.

The full scientific/engineering requirements remain in `IMPLEMENTATION_PLAN.md` and take precedence over this summary.
