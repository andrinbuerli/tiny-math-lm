# Compute-matched continued-SFT plan

The current Experiment 15 comparison gives continued SFT 100,000 optimizer
updates while GRPO receives 10,000. The old continued-SFT metric also subtracts
source-SFT tokens even though its counter starts at zero. Its reported compute
ratios therefore overstate the mismatch in an inconsistent way and should not
be treated as valid matched-budget evidence.

## Budget and stopping rule

Train a fresh continued-SFT optimizer from each verified Experiment 14 SFT
checkpoint. Match the completed Experiment 15 GRPO arm by estimated dense
training FLOPs, using `6 × parameter_count × padded input positions` for the
continued-SFT updates. Padded and prompt positions count; source SFT, validation,
qualitative generation, and final evaluation do not.

| Model | GRPO target FLOPs | Planning estimate for SFT updates |
|:---|---:|---:|
| 1M | 6,290,593,446,703,104 | 19,867 |
| 10M | 49,035,859,649,187,840 | 15,432 |
| 100M | 434,305,201,066,598,400 | 13,767 |

The update counts are estimates only. Stop after the first complete optimizer
update that reaches the target; the actual padded batch lengths determine the
final count. Require less than 0.1% overshoot. A 100,000-update ceiling is a
safety limit, not a success condition. Use cosine learning-rate decay by FLOP
progress after the 100-update warmup.

## Execution order

1. Verify the source SFT checkpoint hashes, parameterization, tokenizer,
   reasoning-v3 data bundle, seed 18, fixed evaluation population hashes, and
   each GRPO update-10,000 checkpoint/metric record.
2. Use the separate Hydra preset
   `15_reasoning_v3_arithmetic_subtasks_compute_matched`; preserve the existing
   Experiment 13, 14, and 15 presets and results.
3. Train and evaluate one model at a time in order 1M, 10M, then 100M. Before
   loading each model, require working GPU telemetry, utilization at or below
   75%, and free VRAM of at least the model reserve plus 2 GiB (1M: 3 GiB,
   10M: 5 GiB, 100M: 10 GiB). Wait when these limits are not met. Use a fresh
   online W&B identity per control and resume its identity after interruption.
4. Evaluate each final budget checkpoint on the same greedy test,
   number-length OOD, and expression-length OOD pools: 50,000, 5,000, and 2,000
   parent questions respectively.
5. Report target and achieved FLOPs, budget error, updates, checkpoint hashes,
   evaluation identities, and every model/split accuracy. Only after all three
   evaluations pass identity and completeness checks should the README's
   primary comparison and figures switch to the matched controls.

This is a match under the repository's versioned FLOP estimator, not a match of
GPU time or hardware counters. It remains a single-seed comparison and does
not predict whether matching compute will favor either training method.
