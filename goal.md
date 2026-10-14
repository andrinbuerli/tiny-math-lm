# Experiment 13 reasoning-v3 comparison table

## Objective

Produce the originally planned side-by-side accuracy comparison table for Experiment 13, using the reasoning-v3 dataset at `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`. Use the Experiment 13 reasoning-v3 table and compare it with the matching frozen baseline results. Keep this goal scoped to producing and verifying that comparison deliverable.

Preserve source bundles, checkpoints, Experiment 11 artifacts, W&B histories, and unrelated working-tree changes. Do not relabel or overwrite historical artifacts. Work only in `/workspace`; never push, fetch, pull, configure a remote, or contact a Git remote. Follow `AGENTS.md` and `IMPLEMENTATION_PLAN.md`, use `uv` and Hydra for experiment entry points, and use online W&B for every evaluation or comparison run.

## Required table

| Model | Arm | Split | Baseline accuracy (scored) | +Reasoning accuracy (scored) | Δ percentage points | Status |
|---|---|---|---:|---:|---:|---|

Include all model, arm, and split rows specified by the original experiment plan. Use only the Experiment 13 reasoning-v3 table for reasoning-side data. Verify baseline and reasoning scores use identical question IDs before calculating a delta. Calculate `Δ percentage points = 100 × (reasoning accuracy − baseline accuracy)` only when both evaluations are complete and question identities match. For incomplete or mismatched evaluations, leave the delta blank and report the true status: `not-run`, `running`, `partial`, `failed`, or `complete`.

Never invent scores, results, run links, provenance, or completion states. Keep partial and missing baseline results visible. Record per-module accuracy, trace/format validity, output lengths, token/compute costs, experiment identities, and W&B links when those values are available and verified.

## Execution prompt

1. Inspect repository instructions, `IMPLEMENTATION_PLAN.md`, Git status, active TinyMath processes, W&B run metadata, checkpoints, the requested data bundle, its reasoning-v3 table, and the report-generation path. Preserve unrelated changes and historical artifacts.
2. Confirm old-format Experiment 11 jobs are stopped before any incompatible evaluation. Identify processes first; stop only relevant TinyMath jobs, never unrelated work. Record stopped runs, W&B IDs/statuses, final metrics, and last durable checkpoints.
3. Verify `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18` exists and inspect its manifest/table. Confirm source and trace-format provenance, split membership/counts, question identities, hashes, and forward-only reasoning-v3 structure. Do not modify the source bundle or reinterpret old-format data/checkpoints. If the bundle is missing or invalid, stop dependent work and record the exact blocker.
4. Determine which required evaluations already exist and which are missing. Reuse only results whose model, arm, checkpoint, protocol, split, and question identities match. Run missing evaluations with Experiment 13 names and fresh W&B identities, online logging, Hydra-composed sanitized configs, and `uv`. Start jobs only when hardware permits. If authentication, hardware, data, or another external condition prevents a run, mark it `not-run` and state the specific blocker.
5. Generate the comparison report at `reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/intermediate_results.md`, plus the CSV/JSON results and charts required by the original plan. Preserve the planned table schema and show every required row with its real status. Include provenance and W&B links only when verified.
6. Before each local Conventional Commit, run relevant tests, inspect the diff, run `git diff --check`, and stage explicit intended files only. Never commit datasets, checkpoints, weights, credentials, W&B local state, or secrets.

## Completion record

Update this file as work proceeds. At completion record:

- report, CSV/JSON, and chart paths;
- reasoning-v3 bundle/table identity, provenance, row/split counts, and question-ID matching evidence;
- table row coverage and actual baseline/reasoning scores and deltas;
- evaluations started, reused, stopped, failed, or not-run, with W&B IDs and sync statuses;
- actual test commands and results;
- checkpoint/data provenance and any limitations;
- local `git log --oneline` and final `git status --short`;
- confirmation that historical Experiment 11 artifacts and source bundles remain unchanged.

## Status

- [ ] Verify repository instructions, current worktree, active jobs, W&B state, and the requested reasoning-v3 bundle.
- [ ] Verify baseline and reasoning evaluation identity/question overlap.
- [ ] Complete required evaluations or record exact blockers and statuses.
- [ ] Generate and validate the comparison report, CSV/JSON results, and required charts.
- [ ] Record tests, W&B sync, provenance, limitations, local commits, and final working-tree state here.
