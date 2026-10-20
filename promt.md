# Experiment 15 GRPO completion handoff

Experiment 13 is canceled. Do not launch, resume, modify, or spend GPU time on Experiment 13. Preserve its runs, checkpoints, reports, and historical records as they are.

Finish Experiment 15 by completing the existing 10M GRPO run, then running the 100M GRPO arm. The 1M GRPO arm is already complete. Do not repeat it.

## Current run

- Experiment preset: `experiment=15_reasoning_v3_arithmetic_subtasks`
- Namespace: `runs/exp15_reasoning_v3_arithmetic_subtasks`
- Seed: 18
- Bundle: `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`
- GRPO: 10,000 updates, 16 prompts per batch, group size 16, temperature 1.0, max completion 512, learning rate `1e-5`, clip `0.2`, KL `0`
- Evaluation splits: `test`, `number_length_ood`, `expression_length_ood`
- 10M GRPO completed at update 10,000. Its final checkpoint is `runs/exp15_reasoning_v3_arithmetic_subtasks/10m/grpo/seed18/latest.pt`; existing W&B ID `c7a774b40fc8501c` is online, finished, and synced. The coordinator validated this result before proceeding.
- 1M GRPO is complete at step 10,000. Preserve its existing checkpoint and report entries.
- 100M GRPO is active in trainer PID 199566 (uv parent PID 199563) under online W&B ID `43cede28d1b251d5`. At the last verification (2026-10-20 20:13 UTC), training metrics and the durable `latest.pt` checkpoint were verified at update 3,000. The checkpoint was verified to include optimizer state (100 parameter-state entries), GRPO arm identity, sampler v2 plus left padding, source `runs/exp14_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f/100m/sft/seed18/best.pt`, and matching reasoning-v3 bundle/tokenizer provenance. Its resolved config has 10,000 updates, 16 prompts × group size 16, temperature 1.0, max completion 512, and the three fixed evaluation splits. W&B is online/running with sync pending until the run finishes. Its coordinator (PID 178398) and finalizer (PID 192695) are alive. Do not restart these processes or launch a duplicate trainer. The coordinator admitted the run after verifying the SFT source checkpoint hash and 100,000-step provenance. From the run start at 15:09 UTC to checkpoint update 3,000 at 20:13 UTC, the observed average implies about 12 hours remaining; refresh this estimate from new metrics before reporting it as current.

## Execution and reporting

1. Verify process state, saved checkpoint step, resolved config, W&B identity, source-checkpoint hash, and current GPU state. Do not create a duplicate 10M worker or overwrite artifacts.
2. The 10M arm is complete and validated at update 10,000 with synced W&B state; do not rerun it.
3. Let the existing 100M run finish under its current process and W&B ID. Verify its final checkpoint at update 10,000 and W&B online/sync state. Do not bypass the configured GPU admission checks or use CPU as a substitute.
4. Evaluate the 10M and 100M final GRPO checkpoints on the same fixed Experiment 15 question IDs for all three splits. Verify checkpoint, data bundle, tokenizer, and evaluation identities before reporting scores.
5. Refresh the Experiment 15 comparison outputs, including `reports/exp15_reasoning_v3_arithmetic_subtasks/seed18/accuracy_comparison.svg`, PNG, CSV, JSON, and intermediate results. Keep 1M and the existing SFT/continued-SFT comparison arms. Use only verified results and leave incomplete cells marked accurately.
6. Keep W&B online. Load credentials securely from `/workspace/.env` through the repository helper; never print or place secrets in configs. Preserve all prior results and do not contact Git remotes.

Follow `/workspace/AGENTS.md` and `IMPLEMENTATION_PLAN.md`. Preserve unrelated worktree changes. Before committing this prompt update, inspect the diff and run `git diff --check`; stage only `promt.md` and use a Conventional Commit message. Do not commit generated run data, checkpoints, `.env`, or W&B state.

## Final report

Report the verified 10M and 100M checkpoint steps, W&B IDs and sync status, evaluation identities and scores, output report paths, relevant commands/checks, limitations, local `git log --oneline`, and `git status --short`. Declare completion only after both GRPO arms, all evaluations, and comparison outputs are verified.
