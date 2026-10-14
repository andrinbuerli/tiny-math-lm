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
- [ ] Run the complete repository test suite; collection currently fails because `/workspace/scripts/run_all.py` is absent.
- [ ] Run the fresh W&B-tracked Experiment 12 annotation job and generate the complete bundle.
- [ ] Verify bundle counts, hashes, five-module coverage, split consistency, token limits, and provenance.
- [ ] Record final W&B sync status, validation results, limitations, and local commit in this file.

## Current execution constraint

The repository is checked out at `/home/abl/workspace/math-llm`, while its runtime/configuration and the provided workspace instructions require `/workspace`. `/workspace` is absent in the current environment. The full source annotation and verifier passes succeeded in memory, but the bundle, tokenizer artifact and W&B annotation run cannot be created from this checkout without the required mount. W&B authentication and sync status have not been tested because no run was started.
