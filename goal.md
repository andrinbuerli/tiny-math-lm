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
