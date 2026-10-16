# Experiment 13 continuation handoff

Continue the Experiment 13 reasoning-v3 comparison in `/workspace`. The user's latest instruction is to run **all continued-SFT arms first, then all GRPO arms**. Continue toward the full 27-row comparison in the attached handoff at `/home/codex/.codex/attachments/10393045-e443-4b49-bc83-a9f8dbc56b1b/pasted-text-1.txt`; reread it before acting. Do not treat the old GRPO-slowdown investigation prompt as the active task.

## Current state

- Namespace: `exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f-eval1k`.
- Bundle: `data/processed/exp13_reasoning_v3_arithmetic_subtasks-seed18`.
- Expected `manifest.json` SHA256: `5d493a0f4c961b84b873036ea1bc93ebb6d01b6f022d7ed81997b094da3a018f`.
- The selected SFT `best.pt` checkpoints for 1m, 10m, and 100m are present at 100,000 steps under `runs/<namespace>/<model>/sft/seed18/best.pt`.
- No continued-SFT arm had started at the last verified check.
- The 1m and 10m GRPO workers and the coordinator were interrupted; no matching processes remained. Resume these two runs from their existing durable checkpoints and W&B identities. Do not overwrite or delete any artifacts.

| Model | Durable GRPO checkpoint | Last metrics row at interruption | Existing W&B run ID |
|---|---:|---:|---|
| 1m | 3,000 | 3,396 | `28beded9b3caced0` |
| 10m | 3,500 | 3,737 | `daf32570fe9d756e` |
| 100m | No GRPO checkpoint/run | — | Not started |

The GRPO runner resumes from `latest.pt` and truncates `metrics.csv` after the resumed checkpoint step. Preserve `tracking.resume=true` so 1m and 10m continue under their existing W&B run IDs. The 100m GRPO should receive a new W&B identity in this namespace.

## Blocking condition and launch gate

At the latest check, `nvidia-smi` exited 255 with `Failed to initialize NVML: Unknown Error`; PyTorch reported CUDA unavailable and zero devices. The user has now independently reproduced the same `nvidia-smi` error. No training process is currently running.

Do not launch training until `nvidia-smi` and PyTorch CUDA initialization both succeed and the configured GPU utilization/free-VRAM admission limits can be verified. Do not bypass the gate, run the 100k-update jobs on CPU, change Docker/host configuration, or leave a blind coordinator polling indefinitely. Record the exact evidence and stop at this blocker if GPU access is still unavailable.

## Required execution order after GPU recovery

1. Recheck the worktree, process list, bundle manifest/hash, CUDA, and W&B authentication. Preserve existing user edits, including the current changes in `src/tinymath/generation.py` and `src/tinymath/train/grpo_runner.py`; inspect them before changing any source.
2. Run continued-SFT for **1m, 10m, and 100m**, each initialized from that model's selected `sft/seed18/best.pt`. Use the `continued_sft` Hydra trainer and the existing namespace, seed 18, canonical bundle, online W&B, and its per-model continued-SFT identity. Select the continued-SFT checkpoint by validation loss and evaluate it on the three required splits.
3. Only after all three continued-SFT arms finish, resume GRPO for **1m and 10m** from their existing `grpo/seed18/latest.pt`, maintaining W&B IDs `28beded9b3caced0` and `daf32570fe9d756e`. Start the 100m GRPO arm from its corresponding new SFT `best.pt` with a new W&B run. Evaluate each GRPO latest checkpoint on the same test, number-length-OOD, and expression-length-OOD IDs.
4. Keep W&B online and load credentials from `/workspace/.env` through the repository helper. Never print, copy, track, upload, or put secrets in configs. Ensure every launched run logs provenance and a qualitative sample before proceeding (or immediately if the configured cadence has not yet produced one).
5. Regenerate the 27-row Experiment 13 report at `reports/exp13_reasoning_v3_arithmetic_subtasks/seed18/` only from verified, matching IDs. Keep incomplete deltas blank and statuses evidence-based. Preserve all historical runs/checkpoints and do not contact Git remotes.

The existing `scripts/run_all.py` executes GRPO before continued-SFT by default. Do not run it with both arms enabled if that would violate the requested order. Either use two controlled orchestration phases (continued-SFT enabled and GRPO disabled, then GRPO enabled and continued-SFT disabled) or invoke the Hydra child entry points in a controlled sequence. Inspect the exact resolved overrides and run identities first; avoid accidentally rerunning or replacing completed SFT/W&B artifacts.

## Repository requirements

Follow `/workspace/AGENTS.md` and `IMPLEMENTATION_PLAN.md`. Use `./.tools/uv run --locked` and Hydra for experiment/evaluation entry points. Commit only intended source/config/test/docs paths after inspecting diffs and running relevant checks plus `git diff --check`. Preserve unrelated staged/unstaged user changes. Never commit data, checkpoints, weights, run outputs, `.env`, `wandb/`, or secrets. Never push, fetch, pull, configure a remote, or contact a Git remote.

## Final status to report

Report actual process state, per-arm status and checkpoint steps, W&B run IDs and sync status, evaluation/report completion, exact test/check commands, limitations, local `git log --oneline`, and `git status --short`. Do not claim a launch, result, completion, or W&B sync without evidence.
