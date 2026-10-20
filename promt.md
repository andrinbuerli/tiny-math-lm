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
- 10M GRPO is active in PID 179095 under existing W&B ID `c7a774b40fc8501c`; continue this exact process and identity. At the last inspection, the metrics file had reached update 6,938 and the latest durable checkpoint was step 6,500. Its coordinator (PID 178398) and finalizer (PID 192695) are also alive. Do not restart either process or launch a duplicate trainer; let the current coordinator validate 10M completion and proceed to 100M.
- 1M GRPO is complete at step 10,000. Preserve its existing checkpoint and report entries.
- 100M has not started. Initialize it from the verified Experiment 14 100M SFT `best.pt` referenced by the Experiment 15 Hydra preset. Create a new online W&B identity for this arm.

## Execution and reporting

1. Verify process state, saved checkpoint step, resolved config, W&B identity, source-checkpoint hash, and current GPU state. Do not create a duplicate 10M worker or overwrite artifacts.
2. Let the current 10M run finish and verify its final checkpoint at update 10,000 and its W&B online/sync state.
3. Run 100M only after 10M completes and its provenance validates. Use the existing Experiment 15 Hydra preset and `./.tools/uv run --locked`; preserve the preset's admission checks for GPU utilization and free VRAM. Never bypass hardware admission or use CPU as a substitute.
4. Evaluate the 10M and 100M final GRPO checkpoints on the same fixed Experiment 15 question IDs for all three splits. Verify checkpoint, data bundle, tokenizer, and evaluation identities before reporting scores.
5. Refresh the Experiment 15 comparison outputs, including `reports/exp15_reasoning_v3_arithmetic_subtasks/seed18/accuracy_comparison.svg`, PNG, CSV, JSON, and intermediate results. Keep 1M and the existing SFT/continued-SFT comparison arms. Use only verified results and leave incomplete cells marked accurately.
6. Keep W&B online. Load credentials securely from `/workspace/.env` through the repository helper; never print or place secrets in configs. Preserve all prior results and do not contact Git remotes.

Follow `/workspace/AGENTS.md` and `IMPLEMENTATION_PLAN.md`. Preserve unrelated worktree changes. Before committing this prompt update, inspect the diff and run `git diff --check`; stage only `promt.md` and use a Conventional Commit message. Do not commit generated run data, checkpoints, `.env`, or W&B state.

## Final report

Report the verified 10M and 100M checkpoint steps, W&B IDs and sync status, evaluation identities and scores, output report paths, relevant commands/checks, limitations, local `git log --oneline`, and `git status --short`. Declare completion only after both GRPO arms, all evaluations, and comparison outputs are verified.
