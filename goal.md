# Experiment 13 reasoning-v3 comparison table

## Objective

Produce the originally planned side-by-side accuracy comparison table for Experiment 13, using the reasoning-v3 dataset at `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`. Use the Experiment 13 reasoning-v3 table and compare it with the matching frozen baseline results. Keep this goal scoped to producing and verifying that comparison deliverable.

Preserve source bundles, checkpoints, Experiment 11 artifacts, W&B histories, and unrelated working-tree changes. Do not relabel or overwrite historical artifacts. Work only in `/workspace`; never push, fetch, pull, configure a remote, or contact a Git remote. Follow `AGENTS.md` and `IMPLEMENTATION_PLAN.md`, use `uv` and Hydra for experiment entry points, and use online W&B for every evaluation or comparison run.

## Required table

| Model | Arm | Split | Baseline accuracy (scored) | +Reasoning accuracy (scored) | Δ percentage points | Status |
|---|---|---|---:|---:|---:|---|

## Experiment matrix

Run the three planned model sizes, each with all three training arms. This is nine distinct training runs and 27 model/arm/split evaluation rows. Use fresh W&B run identities for all Experiment 13 runs.

| Experiment 13 run | Model size | Method | Initialization | Training data / objective | Checkpoint used for evaluation |
|---|---:|---|---|---|---|
| `exp13-reasoning-v3-1m-sft-seed18` | 1M parameters | SFT | Fresh random initialization | Reasoning-v3 SFT examples; completion-only next-token loss | Validation-loss-selected checkpoint |
| `exp13-reasoning-v3-1m-grpo-seed18` | 1M parameters | GRPO / RLVR | The selected 1M SFT checkpoint | Prompts and verifiable arithmetic rewards from the RLVR split; no held-out reasoning traces or teacher forcing | Latest GRPO checkpoint |
| `exp13-reasoning-v3-1m-continued-sft-seed18` | 1M parameters | Continued SFT control | The same selected 1M SFT checkpoint used to initialize GRPO | Reasoning-v3 SFT examples; completion-only next-token loss | Validation-loss-selected checkpoint |
| `exp13-reasoning-v3-10m-sft-seed18` | 10M parameters | SFT | Fresh random initialization | Reasoning-v3 SFT examples; completion-only next-token loss | Validation-loss-selected checkpoint |
| `exp13-reasoning-v3-10m-grpo-seed18` | 10M parameters | GRPO / RLVR | The selected 10M SFT checkpoint | Prompts and verifiable arithmetic rewards from the RLVR split; no held-out reasoning traces or teacher forcing | Latest GRPO checkpoint |
| `exp13-reasoning-v3-10m-continued-sft-seed18` | 10M parameters | Continued SFT control | The same selected 10M SFT checkpoint used to initialize GRPO | Reasoning-v3 SFT examples; completion-only next-token loss | Validation-loss-selected checkpoint |
| `exp13-reasoning-v3-100m-sft-seed18` | 100M parameters | SFT | Fresh random initialization | Reasoning-v3 SFT examples; completion-only next-token loss | Validation-loss-selected checkpoint |
| `exp13-reasoning-v3-100m-grpo-seed18` | 100M parameters | GRPO / RLVR | The selected 100M SFT checkpoint | Prompts and verifiable arithmetic rewards from the RLVR split; no held-out reasoning traces or teacher forcing | Latest GRPO checkpoint |
| `exp13-reasoning-v3-100m-continued-sft-seed18` | 100M parameters | Continued SFT control | The same selected 100M SFT checkpoint used to initialize GRPO | Reasoning-v3 SFT examples; completion-only next-token loss | Validation-loss-selected checkpoint |

Evaluate every run on the same three held-out splits: `test`, `number_length_ood`, and `expression_length_ood`. Each evaluation uses the matching Experiment 13 reasoning-v3 table and the corresponding frozen baseline rows. Each model/method pair must report baseline and reasoning scored counts and accuracy, question-ID match status, delta in percentage points when eligible, and a row status. The table therefore contains one row per run per split.

Run order is by model size: for each size, train and evaluate SFT first; use its selected checkpoint to initialize GRPO and continued SFT; then evaluate those two arms. The three sizes may run in parallel only when GPU utilization and VRAM permit it. All training and evaluation runs must log online to `andi-mueller-csem-sa/tinymath-rl`, with Experiment 13 names, groups, tags, and sanitized resolved Hydra configurations.

Use seed 18 and the budgets and optimization settings in the original Experiment 11 plan unless the Experiment 13 Hydra configuration explicitly defines a compatible replacement. The planned defaults are 100,000 updates per arm; SFT effective batch 32; GRPO eight prompts by eight completions, temperature 0.8, learning rate `1e-5`, clip 0.2, and zero KL coefficient; continued SFT effective batch 128 and learning rate `1e-4`, with approximate compute matching reported from measured FLOPs. Select SFT and continued-SFT checkpoints by validation loss and evaluate GRPO's latest checkpoint. Do not use held-out reasoning traces as prompts or RLVR teacher forcing.

For each model/arm, evaluate the same question IDs on all three splits. The baseline side must use the corresponding frozen baseline result and the reasoning side must use Experiment 13's reasoning-v3 table/protocol. Record the checkpoint selector, checkpoint hash, source arm, evaluation identity, and scored count so each of the 27 rows is auditable. If the Experiment 13 Hydra config differs from a listed planned default, record the resolved value and provenance; never silently change budgets or initialization.

Include all model, arm, and split rows above. Use only the Experiment 13 reasoning-v3 table for reasoning-side data. Verify baseline and reasoning scores use identical question IDs before calculating a delta. Calculate `Δ percentage points = 100 × (reasoning accuracy − baseline accuracy)` only when both evaluations are complete and question identities match. For incomplete or mismatched evaluations, leave the delta blank and report the true status: `not-run`, `running`, `partial`, `failed`, or `complete`.

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

- [x] Verify repository instructions, current worktree, active jobs, W&B state, and the requested reasoning-v3 bundle.
- [x] Verify baseline and reasoning evaluation identity/question overlap.
- [ ] Complete required evaluations or record exact blockers and statuses.
- [ ] Generate and validate the comparison report, CSV/JSON results, and required charts.
- [ ] Record tests, W&B sync, provenance, limitations, local commits, and final working-tree state here.

## Progress record (2026-10-14 19:52 UTC)

- No TinyMath experiment processes were active during the initial process inspection, so no old-format run was stopped. No Experiment 13 model checkpoints or evaluation outputs existed before the new launch. Historical Experiment 11 files were left untouched.
- The RTX 4090 was initially idle (0% utilization, 41 MiB used). All three Experiment 13 SFT runs have since started concurrently under the orchestrator's GPU utilization/VRAM gate. Current observed steps: 1M 699, 10M 99, 100M 399; the counters will advance while the runs continue.
- W&B project/entity: `andi-mueller-csem-sa/tinymath-rl`; online auth verified without exposing credentials. Training runs are active and currently have pending live sync state:
  - 1M SFT: run `a93c5a0c07a37bbb`, [W&B run](https://wandb.ai/andi-mueller-csem-sa/tinymath-rl/runs/a93c5a0c07a37bbb).
  - 10M SFT: run `acd0e19f9cbab401`, [W&B run](https://wandb.ai/andi-mueller-csem-sa/tinymath-rl/runs/acd0e19f9cbab401).
  - 100M SFT: run `1fbcb5d43712cbaa`, [W&B run](https://wandb.ai/andi-mueller-csem-sa/tinymath-rl/runs/1fbcb5d43712cbaa).
- Frozen baseline snapshot: `baselines/exp10_restarted_100k_greedy_restart_2/seed18/snapshot.json`; 27 baseline measurements are available. For each of the 3 splits, parent rows in the reasoning-v3 bundle match all 9 baseline prediction-file identities exactly: 50,000 `test`, 5,000 `number_length_ood`, and 2,000 `expression_length_ood`. Standalone subtask rows are excluded by the evaluator’s parent-only population.
- Bundle: `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`; manifest SHA256 `d13993417ebdaa0e75665f1aac6d2e2c45e7f8c27564a198b65344e3ea9665d5`; protocol `reasoning_v3`; annotation SHA256 `bc53869610e445a67022140a5c984d6b3d2e43467ae165b7109f3d5207f9c0aa`.
- A first orchestrator attempt exited during Hydra composition because the Experiment 13 config omitted `ood_examples_per_module`; it created no training W&B run or checkpoint. The config was fixed and child composition revalidated before the successful launch.
- Partial report artifacts have been generated under `reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/`: `intermediate_results.md`, `comparison.csv`, `comparison.json`, and `accuracy_comparison.png/.svg`. All 27 rows currently have baseline scores and `not-run` reasoning status because training/evaluation has not completed. The comparison W&B run is online and synced at `6e13b3b7addee8be`; its available-evaluation summary is corrected to 0.
- Targeted validation: `uv run --locked pytest tests/test_exp13_reporting.py tests/test_exp11_reporting.py tests/test_reporting.py -q` — 15 passed. Hydra composition was checked for `run_all.py` and `plot.py`; `git diff --check` passed before implementation commits.
- Implementation commits: `ec25586` adds the Experiment 13 Hydra preset; `b4b5b45` fixes its child-config OOD field; `c327c4e` adds the comparison report and tests. Latest observed unrelated untracked files remain `.devcontainer/`, `AGENTS.md`, and `IMPLEMENTATION_PLAN.md`.
- Follow-up at 2026-10-14 19:56 UTC: all three SFT runs remain active online (1M 1,399 steps; 10M 699; 100M 665). GPU load is 100%, with about 1.8 GiB free; no further workloads have been added at this utilization. The report now includes a measured-FLOPs control-match table, currently pending until GRPO and continued-SFT run.
- A child-config audit found the original orchestrator did not forward the parent reasoning-v3 rollout settings. SFT itself does not use that executor, so current SFT objectives are unaffected; the saved SFT configs show `rollout.executor=none`. Experiment 13 configs now select `v3_exact` directly, and `run_all.py` forwards rollout settings to child Hydra configs. Child and root compositions resolve to `rollout.executor=exact`; GRPO/evaluation have not started yet. The specific orchestration tests pass (29 passed), and the Experiment 13/reporting suite passes (15 passed).
- Additional local implementation commits: `529fbac` propagates the reasoning-v3 rollout and recognizes Experiment 12 as annotation-only in the numbering invariant; `4acde09` adds compute-matching reporting. The updated partial comparison W&B run (`6e13b3b7addee8be`) synced online with 27 rows and 0 completed evaluations.
- Bundle row audit: each SFT run reads 500,000 parent training rows plus 587,700 eligible standalone subtasks (1,087,700 eligible rows total); the remaining subtask rows are marked ineligible. GRPO uses only the 25,000 parent RLVR prompts, while the 40,064 RLVR subtasks remain excluded from prompt sampling. These counts come from the bundle records and the current training loaders.
- Follow-up at 2026-10-14 19:59 UTC: the same three SFT runs remain live in W&B and the workspace (`running`; W&B API verified). Latest observed steps: 1M 1,777; 10M 1,067; 100M 799. The GPU remains near saturation with stable 1.8 GiB free, so the orchestration gate is not adding new concurrent work. Reasoning-v3 generator/verifier tests also passed: `uv run --locked pytest tests/test_reasoning_annotation.py tests/test_reasoning_v3_bundle.py tests/test_v3_rollouts.py -q` — 64 passed.

- Follow-up at 2026-10-14 20:04 UTC: all three fresh SFT runs remain active locally and in the online W&B project/group. Local metrics are at steps 2,499 (1M), 1,799 (10M), and 1,099 (100M); W&B API reports `running` with steps 2,490, 1,790, and 1,090, respectively. The RTX 4090 is at 98% utilization with 1.8 GiB free, so no additional GPU work has been admitted. The full SFT validation at step 1,000 completed for 100M and training resumed. GRPO and continued-SFT remain pending their selected SFT checkpoints.
- Follow-up at 2026-10-14 20:06 UTC: all three concurrent SFTs advanced to local steps 2,699 (1M), 1,999 (10M), and 1,199 (100M). W&B API confirms each remains `running` and online at steps 2,690, 1,990, and 1,190. GPU utilization is 98% with 1.8 GiB free; dependent evaluations, GRPO, and continued-SFT runs have not started.
- Follow-up at 2026-10-14 20:07 UTC: the parallel SFT runs continue to advance, at local steps 2,899 (1M), 2,199 (10M), and 1,299 (100M). W&B API confirms `running` at steps 2,890, 2,110, and 1,290. RTX 4090 utilization remains 99% with 1.8 GiB free, and dependent arms remain queued by the GPU gate.
- Follow-up at 2026-10-14 20:09 UTC: all SFT trainers remain active and the 100M run passed its step-1,300 validation boundary. Local metrics now show steps 3,152 (1M), 2,399 (10M), and 1,399 (100M); W&B reports each run `running` at steps 3,090, 2,390, and 1,390. GPU utilization is 99% with 1.8 GiB free, so downstream runs remain queued.
- Follow-up at 2026-10-14 20:11 UTC: SFT metrics advanced to local steps 3,399 (1M), 2,599 (10M), and 1,499 (100M); each also completed its latest periodic validation. W&B API reports `running` at steps 3,390, 2,590, and 1,490. GPU utilization is 98% with 1.8 GiB free. The orchestrator has not started evaluations, GRPO, or continued-SFT because the source SFT runs have not completed and the GPU is saturated.
- Follow-up at 2026-10-14 20:13 UTC: all three SFT jobs remain active; the 100M job completed validation at step 1,500 and resumed. Local steps are 3,699 (1M), 2,899 (10M), and 1,599 (100M); W&B API reports `running` at 3,690, 2,890, and 1,590. GPU utilization is 98% with 1.8 GiB free; no dependent training or evaluation has been admitted.
- Follow-up at 2026-10-14 20:14 UTC: local SFT steps are 3,899 (1M), 3,099 (10M), and 1,699 (100M); W&B API confirms all three runs `running` at steps 3,890, 3,090, and 1,690. The 100M run completed its step-1,600 validation. GPU utilization is 97% with 1.8 GiB free; no SFT arm has completed, so dependent evaluations and training remain pending.
- Follow-up at 2026-10-14 20:16 UTC: local SFT steps reached 4,099 (1M), 3,299 (10M), and 1,799 (100M). W&B API reports the runs `running` at steps 4,090, 3,290, and 1,790. The 100M model completed its step-1,700 validation. GPU utilization is 97% with 1.8 GiB free; downstream arms are still pending.
- Follow-up at 2026-10-14 20:18 UTC: local SFT metrics are at steps 4,299 (1M), 3,599 (10M), and 1,894 (100M); W&B API reports all runs `running` at steps 4,290, 3,590, and 1,860. GPU utilization is 99% with 1.8 GiB free. The 100M run completed its step-1,800 validation and has resumed; the three SFT runs are still the only active Experiment 13 training arms.
- Follow-up at 2026-10-14 20:19 UTC: local metrics reached 4,599 (1M), 3,799 (10M), and 1,989 (100M); W&B reports the corresponding online runs `running` at 4,590, 3,790, and 1,950. The 100M model passed its step-1,900 validation. Durable `latest.pt` checkpoints exist for all three SFT arms (1M and 10M refreshed this minute; 100M last refreshed at 20:12 UTC). GPU utilization is 99% with 1.8 GiB free; no dependent run has started.
- Follow-up at 2026-10-14 20:21 UTC: local SFT metrics are at steps 4,699 (1M), 3,899 (10M), and 1,999 (100M). W&B API confirms all three runs `running`, at steps 4,690, 3,910, and 1,990. The RTX 4090 remains at 99% utilization with 1.8 GiB free. No SFT run is complete yet, so the orchestrator has not begun evaluation, GRPO, or continued-SFT.
- Follow-up at 2026-10-14 20:22 UTC: local SFT steps are 4,899 (1M), 4,099 (10M), and 2,099 (100M); W&B API confirms each run `running` at steps 4,890, 4,090, and 2,090. The RTX 4090 is at 99% utilization with 1.8 GiB free. None of the SFT runs is complete, so no downstream evaluation or GRPO/continued-SFT job has started.
- Follow-up at 2026-10-14 20:25 UTC: the 1M run completed its 5,000-step validation and qualitative probe (saved validation loss 0.12948; probe accuracies were train 0.0, validation 0.0, test 0.1). All three SFT runs are still active: local metrics are 5,099 (1M), 4,521 (10M), and 2,299 (100M), with W&B online steps 5,090, 4,490, and 2,290. GPU utilization is 99% with 1.8 GiB free, so dependent arms remain unstarted.
- Follow-up at 2026-10-14 20:26 UTC: local SFT metrics reached 5,299 (1M), 4,699 (10M), and 2,328 (100M); W&B API confirms online runs `running` at 5,290, 4,690, and 2,320. The RTX 4090 is fully utilized (100%) with 1.8 GiB free. The selected SFT checkpoints and evaluations are not complete yet, so downstream Experiment 13 arms have not started.
