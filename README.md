# TinyMath-RL coding-agent instruction package

This package contains a **plan and agent instructions**, not a completed training repository.

- `IMPLEMENTATION_PLAN.md` — full research specification, uv/Hydra/W&B requirements and incremental local-Git workflow.
- `AGENTS.md` — compact persistent agent instructions to place at the repository root.

Copy both files into `/workspace`, where the coding agent will implement the repository. Keep the existing `/workspace/.env` there; it contains the W&B API key and **must remain local, secret and gitignored**. Do not include the `.env` in prompts, archives or commits.

The agent must manage dependencies with `uv` and commit `uv.lock`, run experiments from Hydra configurations and overrides, log every experiment and evaluation to W&B, create tested incremental **local** commits and never push to a remote. If W&B is unavailable, the agent must explicitly identify offline/unsynced runs rather than claiming they uploaded.

## Implemented repository

The note above records the repository's initial setup. The implementation now lives in this workspace. See [the experiment guide](docs/EXPERIMENT_GUIDE.md) for validated setup, training, evaluation, W&B, and reporting commands.
