# Investigation prompt: GRPO slowdown versus baseline

Investigate why the active Experiment 13 reasoning-v3 GRPO runs appear slower than the completed baseline GRPO experiments. Work read-only: do not edit training code or configs, stop/restart jobs, launch experiments, change the active W&B run identities, bypass GPU admission checks, or modify Docker/host configuration. Do not contact Git remotes. Do not print or expose credentials.

## Runs to compare

Current active runs use the fresh Experiment 13 bundle and run namespace:

- Root: `runs/exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f-eval1k/`
- 1m GRPO: `1m/grpo/seed18/`, W&B ID `28beded9b3caced0`
- 10m GRPO: `10m/grpo/seed18/`, W&B ID `daf32570fe9d756e`
- The corresponding 100m GRPO has not started.

Completed historical baseline runs:

- Root: `runs/restarted_100k_greedy_restart_2/`
- 1m GRPO W&B ID `c8605394b695ef91`
- 10m GRPO W&B ID `3a402a249818cae1`
- 100m GRPO W&B ID `de440b7a315d5b12`

Also inspect the partial reasoning-v1 comparator at `runs/exp11_reasoning_500k_100k/1m/grpo/seed18/` (W&B ID `c4a008c309a5e59c`). It reached only about 978 updates, so treat it as a limited comparator rather than a completed benchmark.

Start by rechecking the live process state, newest metrics, resolved configs, local W&B metadata, and remote W&B run state/history. Current local files and remote run history are authoritative; the snapshot below is only a lead to verify.

## Preliminary evidence to verify

Snapshot from 2026-10-16 around 07:07 UTC (recheck before analysis):

| Run | Progress | Recent W&B interval | Rate from interval | CSV cumulative `wall_seconds` |
|---|---:|---:|---:|---:|
| Current 1m reasoning-v3 | local 3,230; W&B summary 3,220 | 200 updates / 1,853.8 s | 0.1079 updates/s; 9.27 s/update | 34,752.8 s |
| Current 10m reasoning-v3 | local 3,543; W&B summary 3,540 | 200 updates / 1,715.0 s | 0.1166 updates/s; 8.57 s/update | 33,694.0 s |
| Baseline 1m | 100,000 | full-run CSV wall time | about 0.1005 s/update | 10,050.3 s |
| Baseline 10m | 100,000 | full-run CSV wall time | about 0.3644 s/update | 36,436.1 s |
| Baseline 100m | 100,000 | full-run CSV wall time | about 0.5605 s/update | 56,047.4 s |
| Partial Experiment 11 1m reasoning-v1 | 978 | partial-run CSV wall time | about 7.55 s/update | 7,385.3 s |

These numbers show a large apparent gap against the completed 256-context/128-completion baseline: approximately 92x for 1m and 24x for 10m using the current recent interval rates. The partial 512-context/512-completion reasoning-v1 run is much closer to current speed, at roughly 7.55 s/update. Treat this as an important lead, not a causal conclusion: current and baseline protocols, context/completion caps, bundles, concurrency, software revisions, and measurement windows differ. The current `metrics.csv` cumulative `wall_seconds` may also include restored elapsed time; prefer differences between multiple timestamped recent steps when comparing sustained throughput.

The current resolved configs indicate reasoning-v3, context length 512, maximum completion length 512, eight prompts per update, and group size eight. The completed historical baseline configs used context length 256 and maximum completion length 128, also with eight prompts and group size eight. The partial Experiment 11 run used context 512 and maximum completion length 512 but reasoning-v1. The baseline runs use a different bundle/manifest from the current fresh run. These differences make raw seconds-per-update comparisons non-equivalent; measure them, but do not attribute causality from config differences alone.

## Investigation

1. Establish reliable throughput for each active run and each historical comparator. Report update/s and seconds/update from recent timestamped intervals, plus tokens/s and generated/policy/executor tokens per update where available. Include multiple windows to show variance and any pauses. Distinguish wall-clock throughput from cumulative training-time counters, logging cadence, checkpoint/evaluation/qualitative overhead, and idle time.
2. Compare matched model sizes first (1m-to-1m, 10m-to-10m, and 100m-to-100m). Identify all material config and provenance differences: model shape, context length, completion cap and actual completion lengths, prompts/group size, temperature, rollout/scorer/executor versions, trace protocol, data bundle, effective batch, precision, generation/cache path, checkpoint/save cadence, eval cadence, qualitative sampling, logging cadence, software/Git revision, and hardware/concurrency.
3. Inspect local source and local Git history for changes between the baseline and current runs, focusing on GRPO rollout/generation, scoring/verifier/executor, backward pass, metric collection, W&B logging, checkpointing, qualitative examples, and evaluation. Do not assume logging is the cause; quantify its overhead from code paths and timestamps if possible.
4. Break down where time is spent per update if existing logs or instrumentation support it: sampling/generation, exact executor/verifier/scoring, old/current log probabilities, loss/backward/optimizer, W&B/CSV logging, checkpoint saves, and periodic evaluation/qualitative generation. If the current artifacts cannot distinguish these, say exactly which timing evidence is missing and propose the smallest safe read-only or short diagnostic needed later. Do not add instrumentation or run a benchmark in this task.
5. Check whether the active 1m and 10m runs overlap on one GPU and whether that causes contention. Inspect current telemetry and process evidence. NVML has recently failed to initialize; do not infer utilization/free VRAM from CPU usage or device-node visibility, and do not bypass the configured admission guard.
6. Separate expected work growth from regression. Estimate whether the larger context/completion limits, longer generated reasoning traces, exact v3 call execution, and concurrent jobs explain some measured cost. Verify actual token lengths and executor-call counts rather than using configured maxima as if every completion reached them.

## Required output

Return a concise report with:

- A table of run, model, protocol, bundle, update range, measured interval, updates/s, seconds/update, tokens/s, and confidence/limitations.
- Which comparisons are apples-to-apples and which are confounded.
- The strongest evidence-backed explanation(s), ranked by confidence, with paths/metric names or W&B links.
- Whether recent logging-related code changes plausibly account for a material share of the slowdown, and the evidence for that judgment.
- Any unexplained regression after accounting for context/completion length, token counts, rollout/scoring work, and GPU concurrency.
- A minimal recommended follow-up experiment or instrumentation plan, including predicted measurable outcomes. Keep it as a recommendation; do not execute it.

Do not change the running experiment or report just to make this investigation easier. Preserve all current and historical checkpoints, metrics, W&B identities, and user work.
