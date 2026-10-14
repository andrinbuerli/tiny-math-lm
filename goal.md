# Goal: Experiment 12 forward scratchpad correction

## Objective

Create a new, versioned Experiment 12 annotation bundle using `reasoning_v2` forward derivations across the five selected mathematics modules. A trace must begin with derivation steps, contain no operation-summary lines, and omit the final scalar; the separate `A:` completion field carries the answer. Preserve all Experiment 11 data, checkpoints, and run records.

This goal is data-only. Do not stop, start, restart, or resume model training, and do not launch evaluations or retrain checkpoints. The only permitted experiment run is the fresh, W&B-tracked Experiment 12 annotation run.

## Requirements

- Preserve `reasoning_v1` parsing, tokenizer, and checkpoint compatibility for historical bundles.
- Independently validate `reasoning_v2` ordering and arithmetic/equation replay, including budget-reduced traces.
- Keep the strict one-Unicode-code-point-per-ordinary-token tokenizer and answer-after-scratchpad completion format.
- Derive the new bundle from the existing source bundle without modifying it. Record trace version, source manifest/split hashes, tokenizer hash, annotation hash, row counts, module coverage, and split provenance.
- Store the generated dataset under `data/processed/exp12_reasoning_forward_500k_100k-seed18`; never write into Experiment 11 output paths.

## Status

- [x] Confirm no active TinyMath training or annotation process before implementation.
- [x] Add versioned forward trace generation, parsing, and verification paths.
- [x] Add Experiment 12 annotation configuration and this goal record.
- [x] Run focused tests and lint; all reasoning v1/v2 focused tests pass.
- [x] Annotate all 607,000 source rows in memory; all traces fit the 512-token ceiling.
- [x] Independently verify/replay all 607,000 generated traces in memory across all five modules.
- [x] Attempt the complete repository test suite; pytest collection stops because `/workspace/scripts/run_all.py` is absent.
- [x] Run the fresh W&B-tracked Experiment 12 annotation job and generate the complete bundle.
- [x] Verify bundle counts, hashes, five-module coverage, split consistency, token limits, and provenance.
- [x] Record validation results, workspace limitation, and W&B status in this file.

## Current execution constraint

The repository is checked out at `/home/abl/workspace/math-llm`, while its default runtime root is `/workspace`, which is absent in this environment. The annotation entry point now honors `WORKSPACE_ROOT`; setting it to this checkout allowed the new bundle and run artifacts to be written here without changing Docker or host configuration. The Experiment 11 output remains untouched.

Validation: 607,000 rows written and checked; 500,000 `sft_train`, 25,000 `sft_validation`, 25,000 `rlvr`, 50,000 `test`, 5,000 `number_length_ood`, 2,000 `expression_length_ood`, and 0 `extrapolate_other`. Every split matched its source rows and recorded hashes; all five modules are covered. Every trace is `reasoning_v2`, starts with a derivation step, has no `A:` field, and fits within 512 tokens. Tokenizer and annotation digests were verified from the saved artifacts.

W&B annotation run `e96ed0e74ca4078f` finished online and synced (`exit_code=0`): https://wandb.ai/andi-mueller-csem-sa/tinymath-rl/runs/e96ed0e74ca4078f. No training or evaluation jobs were started. The complete repository test suite still cannot collect because `/workspace/scripts/run_all.py` is absent; focused tests pass.

Implementation commit: `6aa7243 feat(data): add forward reasoning v2 annotations`. Workspace-root fix and final status are recorded in the follow-up commit.

## Linear trace completeness follow-up

The first `reasoning_v2` bundle omitted the divide and substitution-verification steps from linear-equation traces. The converter now preserves the collect and move steps, shows division by the isolated coefficient, and keeps a nonnumeric verification step while suppressing the explicit solution and numeric substitution result. The corrected full-steps bundle is `data/processed/exp12_reasoning_forward_fullsteps_500k_100k-seed18`, generated as a separate Experiment 12 annotation run; the earlier bundle remains unchanged.

W&B annotation run `83a858ec50c206f6` finished online and synced. It wrote and verified 607,000 rows with the original split counts and reasoning_v2 provenance. No model-training or evaluation runs were started.

# Successor goal: Experiment 13 reasoning-v3 comparison table

This successor request uses **Experiment 13** and the dataset at `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`. It supersedes the earlier Experiment 12 data-only restriction for this new request; Experiment 12 bundles, checkpoints, reports, run metadata, and completion records remain historical and must not be relabeled or overwritten. Preserve unrelated working-tree changes. Work only in `/workspace`; do not push, fetch, pull, configure a remote, or contact a Git remote.

## Objective

Produce the comparison table planned in the original experiment brief using the Experiment 13 bundle and its reasoning-v3 table. First verify the bundle, its reasoning-v3 table, and provenance. If the bundle or table is absent or incompatible, do not fabricate or silently substitute data; record the precise blocker.

## Required comparison table

| Model | Arm | Split | Baseline accuracy (scored) | +Reasoning accuracy (scored) | Δ percentage points | Status |
|---|---|---|---:|---:|---:|---|

Include all model/arm/split rows required by the original plan. Compare baseline and reasoning scores only when they cover identical question IDs. Calculate `Δ percentage points = 100 × (reasoning accuracy − baseline accuracy)` only when both evaluations are complete and question identities match. Leave the delta blank for incomplete or mismatched comparisons. Use only observed statuses: `not-run`, `running`, `partial`, `failed`, or `complete`. Never invent scores, results, run links, or completion states.

## Execution prompt

1. Inspect repository instructions, `IMPLEMENTATION_PLAN.md`, Git state, active TinyMath processes, W&B run metadata, checkpoints, annotated bundles, trace generator/verifier, and report-generation path. Preserve unrelated edits and historical artifacts.
2. Identify any still-running old-format Experiment 11 jobs before stopping them. Stop only those TinyMath jobs; leave unrelated processes alone. Record each stopped run's W&B ID/status, final metrics, and last durable checkpoint. The prior monitoring reported that Experiment 11 GRPO, 1M continued-SFT, 100M SFT, and the queued launcher were stopped; recheck current state instead of relying on that report.
3. Inspect `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18` and its reasoning-v3 table. Verify the manifest, source bundle, trace version, question IDs, split membership, and hashes. Do not alter the source bundle.
4. Verify the reasoning-v3 table contains forward derivations: operands and derivation first, no result-first or redundant result-summary lines, and results/final answers only after supporting steps. Use exact deterministic arithmetic and never Python `eval`.
5. Run the evaluations and comparisons required by the original plan using Experiment 13 naming and fresh W&B run identities. Use Hydra and `uv`; log online to the configured W&B entity/project and save sanitized resolved configs/provenance. Confirm hardware capacity before launching jobs. If authentication, hardware, bundle validation, or another external dependency prevents a run, mark it `not-run` and state the exact blocker.
6. Preserve all old bundles/checkpoints. Treat checkpoints trained on an incompatible trace format as incompatible; do not silently reuse them. Generate the planned report and required CSV/JSON/figures with accurate partial/failure states and verified provenance.
7. Before each local Conventional Commit, run relevant tests, inspect the diff, run `git diff --check`, and stage explicit intended paths only. Never commit datasets, checkpoints, credentials, W&B local state, or secrets.

## Experiment 13 completion record

Update this section as execution proceeds. At completion, record the report path; exact bundle/table identity and question-identity checks; row coverage; actual runs started/stopped/not-started with W&B IDs and sync statuses; test commands and results; bundle/checkpoint provenance; limitations; local `git log --oneline`; and final `git status --short`. Confirm Experiment 11 and Experiment 12 historical artifacts remain unchanged. Do not claim completion, run starts, results, or W&B sync that were not verified.

**Current status:** Prompt recorded. Experiment 13 implementation/evaluation/run status and requested bundle provenance have not been verified in this section.
