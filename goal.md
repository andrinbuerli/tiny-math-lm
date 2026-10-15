# Experiment 13 — Fresh reasoning-v3 comparison handoff

## Goal

Produce the complete Experiment 13 side-by-side accuracy comparison using the **current** bundle at `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`. The deliverable is a 27-row comparison of frozen baseline and reasoning-v3 accuracy, with auditable identities, scores, deltas, and statuses.

**Restart policy: start the Experiment 13 training matrix from scratch. Do not resume or load any checkpoint from the stopped attempt.** Preserve the stopped checkpoints and W&B histories as historical artifacts. Fresh SFT runs start from random initialization on the current bundle; the GRPO and continued-SFT arms start from their corresponding *new* SFT selected checkpoint.

## Required comparison

| Model | Arm | Split | Baseline accuracy (scored) | +Reasoning accuracy (scored) | Δ percentage points | Status |
|---|---|---|---:|---:|---:|---|

Include every combination of:

- Models: `1m`, `10m`, `100m`.
- Arms: `sft`, `grpo`, `continued_sft`.
- Splits: `test`, `number_length_ood`, `expression_length_ood`.

Use seed 18. Use the Experiment 13 Hydra budgets: 100,000 updates per arm; SFT effective batch 32; GRPO 8 prompts × 8 completions, temperature 0.8, learning rate `1e-5`, clip 0.2, zero KL; continued SFT effective batch 128, learning rate `1e-4`. Select SFT and continued-SFT checkpoints by validation loss; evaluate GRPO's latest checkpoint. Report measured FLOPs and approximate compute matching. Do not use held-out reasoning traces as prompts or RLVR teacher forcing.

For each model/arm, evaluate identical question IDs on all three splits. Verify each reasoning-side ID set against the matching frozen baseline IDs before computing `100 × (reasoning accuracy − baseline accuracy)`. Leave deltas blank for incomplete or mismatched evaluations. Report only verified states: `not-run`, `running`, `partial`, `failed`, or `complete`. Include scored counts, per-module accuracy, trace/format validity, output lengths, token/compute costs, checkpoint selector/hash, source arm, evaluation identity, and W&B link when available.

## Clean restart rules

1. Use the current bundle path and verify its manifest and split hashes before training. Do not edit, regenerate, or silently reinterpret the source bundle.
2. Use a distinct run namespace so no checkpoint, metrics file, W&B run ID, or resolved config from the stopped attempt is overwritten. Suggested namespace: `exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f` (the suffix identifies the current bundle manifest). Keep the report at the required original report path by overriding `reporting.output_dir` when generating it.
3. Create fresh online W&B identities in entity `andi-mueller-csem-sa`, project `tinymath-rl`, with names, group, and tags containing the fresh namespace and model/arm. Set `tracking.resume=true` only so an interruption of these *new* runs resumes their own identity. Do not reuse the old run IDs below.
4. Persist sanitized Hydra-resolved configs and full bundle/checkpoint/Git provenance in each new run directory. Use `uv run --locked` and Hydra for every experiment/evaluation entry point. Never log credentials.
5. Start parallel model jobs only when live GPU utilization and free VRAM are verifiable and satisfy the configured admission limits. The current Experiment 13 limits are utilization ≤75%, a 2 GB headroom reserve, and per-model VRAM reservations of 1/3/8 GB for 1M/10M/100M. Otherwise queue work. Do not fall back to CPU for the 100k-update experiment matrix.
6. If GPU, W&B authentication/network, or bundle validation prevents launch, do not claim a run started. Record exact evidence and mark all affected rows `not-run` (or `running` only if an evaluation actually started).

Suggested full-training entry point after GPU recovery (compose and inspect before launching):

```bash
./.tools/uv run --locked python scripts/run_all.py \
  --config-name reasoning_v3 \
  experiment=13_reasoning_v3_arithmetic_subtasks \
  experiment.name=exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f \
  experiment.lineage=exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f \
  orchestration.parallel_models=true \
  tracking.mode=online \
  tracking.project=tinymath-rl \
  tracking.entity=andi-mueller-csem-sa \
  tracking.group=exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f \
  tracking.resume=true \
  data_bundle=data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18
```

The fresh experiment name must route output to new `runs/` paths. Do not remove, rename, or repurpose historical directories. If the configured name/path behavior differs from this expectation, fix the invocation or path configuration before starting.

## Verified handoff state (2026-10-15)

### Current bundle

- Path: `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`.
- `manifest.json` SHA256: `5d493a0f4c961b84b873036ea1bc93ebb6d01b6f022d7ed81997b094da3a018f`.
- Source manifest SHA256: `3a609d313d1025e8801bf7ef63b916ab680ac07a4af33842fb703b3cf27740f0`.
- Trace protocol: `reasoning_v3`; annotation SHA256: `97c89724efc407a4b5d5db417e263259b002377e0bff08c0fcfdc22bca04dcad`; trace budget: 512; strict character tokenizer vocabulary: 84, tokenizer SHA256 `1ec41b1dedab822e57e86a3e6ebd54e9a165431fbe1b1ff9622e89dfed57163d`.
- Selected source questions: 607,000. JSONL row counts and SHA256:

| File | Rows | SHA256 |
|---|---:|---|
| `sft_train.jsonl` | 894,721 | `4900d8b155c791bfccab1f70c4468dbc42714482724661664b659294dd04d3f2` |
| `sft_validation.jsonl` | 44,795 | `a0c61b781aa77617c7057439d8ef74f35373a6b32c207eaa954d414cfa57b767` |
| `rlvr.jsonl` | 44,915 | `11750dfab1aed0aa4a18ebfd17718723d341fb6a21c33104d8a1d2fa6ed3d400` |
| `test.jsonl` | 94,910 | `594aa62a0f4dcb10ea19f72c6c106e67ab8cb277fcfe8f081cc37a6a7094d8c5` |
| `number_length_ood.jsonl` | 8,724 | `f8c830f378f5042a37076c12d2997267337e1f797f4ab1200c5012cd707e3fae` |
| `expression_length_ood.jsonl` | 8,000 | `63b674f6b74e8d355a1b40786bff5075d42e80a00b61fb936deade397c79867c` |
| `extrapolate_other.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

A sample from each nonempty split has a `scratchpad` and `trace_version`; the full verifier, row-level trace validity, parent/subtask counts, and baseline question-ID overlap still need to be rerun against this exact manifest before evaluation. The manifest's `source_bundle` value is provenance metadata only; work from the in-workspace bundle above.

### Stopped attempt — preserve, do not reuse

The earlier attempt used the same path string but a different bundle manifest, and was stopped before any evaluation outputs were produced. Its three SFT jobs were interrupted at local metric steps 82,399 / 79,899 / 35,699 (1M / 10M / 100M). The last durable checkpoints are preserved under `runs/exp13_reasoning_v3_arithmetic_subtasks/{1m,10m,100m}/sft/seed18/latest.pt`:

| Model | W&B ID | W&B state at last API check | Last local step | `latest.pt` SHA256 |
|---|---|---|---:|---|
| 1M | `a93c5a0c07a37bbb` | `crashed` | 82,399 | `27ed4c6bd1c6028aef4e82aea09ebbac857bfa677aa61a734e1fc1cf716254a1` |
| 10M | `acd0e19f9cbab401` | `crashed` | 79,899 | `62d9e445510aa0cf3b0a600d941b658348d5e427ffc62690e410915e03ab033c` |
| 100M | `1fbcb5d43712cbaa` | `crashed` | 35,699 | `2121689e507e7cbfc883eda533e121243370bfc364be30a6a4f08737a16805de` |

The orchestrator and recovery guard were also stopped. A process check found no active TinyMath Experiment 13 jobs. The historical comparison W&B run is `ff900c166e1ef559` and was last verified `finished`/synced at 2026-10-15 06:09 UTC; it describes the previous bundle and must be refreshed for this manifest. No reasoning evaluation result files existed at the last check.

### Hardware blocker — no fresh runs started

At handoff, `nvidia-smi` exits 255 with `Failed to initialize NVML: Unknown Error`. The CUDA-enabled PyTorch build (`2.6.0+cu124`) reports `torch.cuda.is_available() == False`, device count 0, and `torch.cuda.init()` raises `RuntimeError: No CUDA GPUs are available`. NVIDIA device nodes and driver version `580.159.04` are visible, but these do not establish usable GPU capacity. **No new run has been started.** Recheck `nvidia-smi` and CUDA initialization before launching; if still failing, keep runs not started and report this exact blocker. Do not change Docker host configuration.

### Current report/worktree

- Required output directory: `reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/` with `intermediate_results.md`, `comparison.csv`, `comparison.json`, `accuracy_comparison.png`, and `accuracy_comparison.svg`.
- Existing comparison JSON has 27 rows but predates the current manifest; treat it as stale and regenerate after identity validation. Keep every row visible and use blank deltas until both sides match by question ID.
- The most recent repository status check showed this handoff file modified and unrelated untracked `.devcontainer/`, `AGENTS.md`, and `IMPLEMENTATION_PLAN.md`. Preserve those files and any new user changes. Branch is `main`; latest commits include `9768826 fix(data): verify v3 traces with character token budgets` and `f90d83c fix(reasoning): forbid delegation of atomic and whole arithmetic tasks`. No remote Git operations are permitted.

## Completion checklist

- [ ] Recheck worktree, active jobs, current bundle manifest/hashes, CUDA, and W&B before launch.
- [ ] Validate every reasoning-v3 row and verify test/OOD parent question IDs exactly match the frozen baseline IDs.
- [ ] Start all 9 model/arm runs from scratch in the fresh namespace, using only the current bundle; record actual W&B IDs and sync state. If hardware/auth blocks them, record `not-started` and exact evidence.
- [ ] Evaluate all 27 cells with the required checkpoint selectors and matching IDs; compute deltas only for complete matches.
- [ ] Regenerate the Markdown, CSV, JSON, PNG, and SVG report from the current bundle and runs; sync the comparison online to `andi-mueller-csem-sa/tinymath-rl` under a fresh comparison identity.
- [ ] Run relevant validation/tests, inspect diffs, run `git diff --check`, and commit only intended paths with a Conventional Commit. Never commit datasets, checkpoints, weights, `.env`, W&B state, or generated run data.
- [ ] Record actual results, test commands, W&B sync status, provenance, limitations, `git log --oneline`, final `git status --short`, and confirmation that historical Experiment 11 artifacts/source bundles remain unchanged.
