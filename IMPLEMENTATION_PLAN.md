# TinyMath-RL — Repository Implementation & Experiment Plan

## Goal

Build a reproducible, single-GPU research repository comparing **~1M, ~10M, and ~100M parameter decoder-only Transformers** trained from scratch on basic mathematics and subsequently improved with **reinforcement learning with verifiable rewards (RLVR)** using **GRPO**.

Primary question: **How do mathematical accuracy, extrapolation, and RLVR gains scale with model capacity?**

Target hardware: **one NVIDIA RTX 4090 (24 GB)**. All experiments and generated artifacts must remain within the repository workspace (`/workspace` in the Codex GPU Docker container).

## Required tooling and experiment tracking

**Mandatory across the entire repository:**

- **uv** for Python 3.11+ interpreter/environment management, dependency installation, dependency locking and command execution. Maintain `pyproject.toml` and a committed `uv.lock`. Use `uv sync --locked` for reproducible setup and `uv run` for all documented commands. Do not use `pip install`, Poetry, Conda or ad hoc virtualenv management as the project's standard workflow. Work with the existing container's CUDA/PyTorch compatibility; do not modify the Docker host.
- **Hydra + OmegaConf** for all configuration composition and experiment entry points. Use structured YAML config groups (`model`, `data`, `trainer`, `rlvr`, `eval`, `tracking`, `experiment`) with explicit defaults and CLI overrides. Avoid parallel custom argparse/YAML configuration systems. Resolve and save the full composed config for every run; ensure paths stay rooted under `/workspace` despite Hydra's run-directory behavior.
- **Weights & Biases (W&B / `wandb`) is for experiments only**: SFT, GRPO, continued-SFT, named overfit/optimization smoke experiments, 2k/20k pilots, full runs, checkpoint evaluations, OOD benchmarks and experimental comparisons. Every experimental invocation creates or resumes a corresponding W&B run with a unique ID; no silent opt-out.

### Experiment-only tracking scope

- Track scientific experiment metrics, configurations, provenance and results. Data preparation and annotation may be tracked only as stages of a named experiment.
- Documentation/README updates, code edits, refactoring, unit/integration tests, linting, static checks, Hydra configuration validation, local browsing, Git operations and preparing documentation figures must not create, resume or upload W&B runs. Rendering an existing result plot for a README is documentation work.
- Tests must use local fixtures or mocked tracking. Executing an optimizer inside a regression test does not make it an experiment. A Hydra experiment preset alone does not make a maintenance task an experiment.
- Do not load W&B credentials, authenticate, create authentication-only runs or inspect remote run status for routine development work. Verify real tracking as part of an actual experiment when one is launched.
- Include W&B tracking/sync status in user-facing reports only when an experiment was run or diagnosed, or when the user explicitly requests it. Omit W&B status from routine development and documentation summaries.

This scope applies to all W&B requirements, milestones and validation commands below.

### W&B authentication and secret handling

- An existing **local `/workspace/.env`** file contains the W&B API key. **Read it at runtime only when an experiment requires tracking**, using `python-dotenv` or an equivalent secure loader; support `WANDB_API_KEY` in the environment and standard `.env` assignment syntax. Do not assume the file contains anything else. Never display, print, log, commit, copy into Hydra config, or upload the key.
- Never create a checked-in `.env` with credentials. Gitignore `.env`, `.env.*` except an optional secret-free `.env.example`. Do not run a command that exposes secrets in shell history, process arguments, test assertions, tracebacks, or agent responses. Prefer `wandb.login(key=os.environ["WANDB_API_KEY"])` or environment-based implicit login without printing the value.
- Authenticate and verify W&B logging as part of the first actual experiment, using the user's existing workspace credentials; do not create a separate infrastructure/authentication run. If that experiment encounters missing/invalid credentials or unavailable W&B connectivity, **report the specific blocker**; do not claim remote tracking succeeded. You may run experimental smoke runs with `WANDB_MODE=offline` only when required by connectivity and must mark them as unsynced, keep queue files under `/workspace`, and document `uv run wandb sync <offline-run-dir>`; do not silently downgrade normal experiments to offline or disabled logging. Ordinary development checks need no W&B connection or offline run.
- W&B project/entity, run names, groups, tags, logging frequency, artifact policy and tracking mode must be Hydra-configurable. Do not hardcode a personal account name. Default project name `tinymath-rl`, with W&B entity resolved from authenticated account or optional environment/config setting.
- Log full **sanitized** resolved Hydra config, Git SHA, uv lockfile digest, package/CUDA/GPU information, dataset revision, split manifest hash, tokenizer hash, model size, seed, training arm, checkpoint provenance, step/token count and protocol name. Exclude API keys, `.env` contents and any other sensitive fields from config and metadata.
- Log scalar curves (`train/loss`, `val/loss`, accuracy, throughput, wall time, tokens, learning rate, gradient norms, GPU peak memory), GRPO diagnostics (reward, mixed/all-correct/all-wrong groups, entropy, KL, clipping, verifier failure rate), module breakdowns, OOD results, plots and final summaries. Use consistent metric namespaces and meaningful step definitions (`optimizer_step`, `tokens_seen`, `evaluation_step`); define W&B metrics rather than mixing incompatible x-axes.
- Ensure Hydra run IDs and W&B run IDs are persisted locally so interrupted runs can **resume the same W&B run** (`id` + `resume="allow"`/appropriate mode) without accidental duplicates. Finish/flush runs gracefully; keep locally produced metrics CSV/JSON and plots even after successful W&B logging. Small plots/results may be logged as W&B tables/images/artifacts; large raw datasets and model checkpoints stay local by default unless explicitly configured otherwise.
- Every completed evaluation must link via metadata to the evaluated checkpoint, source SFT/GRPO run and relevant data split. Comparative W&B reports/tables must distinguish actual measurements from missing runs; never invent measurements.

## Non-negotiable scope

Use the pre-generated [DeepMind Mathematics Dataset on Hugging Face](https://huggingface.co/datasets/d0rj/mathematics_dataset), restricted to **exactly** these modules:

1. `arithmetic__add_or_sub`
2. `arithmetic__mul`
3. `arithmetic__div`
4. `arithmetic__mixed`
5. `algebra__linear_1d`

No quadratic, calculus, probability, or word-problem datasets. Do not substitute GSM8K.

Use **strict single-character tokenization**: one Unicode character/code point per ordinary token, including each digit, space, operator and letter. No BPE, SentencePiece, grouped digits or pretrained tokenizer. Reserve only minimal special IDs (`PAD`, `BOS`, `EOS`, `UNK`). A character-level rendering of any answer delimiters is required.

## Experimental protocol

### Dataset and splits

- Inspect the actual Hugging Face dataset schema, categories, module identifiers and storage layout before implementing loaders. Do not assume split names.
- Stream/download and filter to the five selected modules. Save a manifest containing dataset revision, module counts, category counts, seeds and preprocessing rules.
- Respect official evaluation and extrapolation partitions when available; do not mix them into training.
- From eligible training examples, create deterministic, **disjoint** SFT train, SFT validation and RLVR question pools. Deduplicate normalized questions across pools and evaluation sets (retain original problem text for model input).
- Treat problem normalization conservatively; do not alter signs, operands or mathematical syntax.
- Balance modules initially and log actual token counts and per-module distributions.
- Default main budget: **500k unique SFT examples** per model, the same examples for all three sizes. Provide 2k-example smoke, 20k pilot, and optional 2M extended presets. Make budgets configurable.
- For RLVR, begin with the disjoint, fixed problem pool. Optionally implement fresh sampling from the original [DeepMind generator](https://github.com/google-deepmind/mathematics_dataset) limited to those five modules, with seeds and provenance recorded.
- Create separate number-length and expression-length OOD tests, preferably using documented original extrapolation splits. Never change multiple difficulty dimensions in a purported single-variable experiment.

### Character tokenizer

- Scan eligible training text and output format to establish the character vocabulary. Persist sorted character-to-ID and ID-to-character mappings with a version/hash.
- All ordinary characters map to **exactly one** token ID; unknown characters are explicit errors during dataset preparation by default (no silent replacement).
- Implement `encode`, `decode`, batching/padding, save/load and round-trip tests.
- Share one frozen tokenizer across 1M/10M/100M, SFT, RLVR and evaluation.
- Inspect distribution of tokenized length (median/p95/p99/max). Begin with context length **256**, increase to **512** only if justified by truncation/exclusion statistics; never silently truncate questions or answers.

### Prompt and completion

Use the original DeepMind question text with a simple, character-tokenized format:

```text
Q: Calculate 23 + 48.
A: 71
```

The model receives the text ending in `A: ` and generates the answer plus EOS. The answer may be an integer, negative number, decimal, fraction, or valid symbolic scalar matching the selected module. Preserve sufficient answer capacity for all module outputs.

Compute supervised cross-entropy **only on completion and EOS**, never on prompt or padding. Verify exact loss-mask boundaries in tests. Keep prompt framing consistent across all experiments.

### Model family

Implement a compact GPT-style autoregressive model with causal SDPA, RoPE, RMSNorm, SwiGLU and tied embeddings where feasible. Define three YAML configurations targeting **1M, 10M, 100M actual trainable parameters**; verify and report counts programmatically. Keep attention, normalization, tokenizer, context length and output representation consistent across sizes. Use bf16 on supported GPUs.

### Stage 1: supervised training (SFT)

- Start from random weights; AdamW, warmup + cosine schedule, grad clipping, gradient accumulation, checkpointing/resume.
- Train all model sizes on identical SFT examples and, for the primary comparison, **equal processed-token budget**. Log examples, effective tokens, wall time, estimated FLOPs and peak VRAM.
- Save checkpoints at fixed token milestones and select according to a predefined validation protocol, not test performance.
- Evaluate using greedy decoding, module-specific accuracy and format validity.

### Stage 2: RLVR with GRPO

Initialize each policy from its SFT checkpoint. For each update:

1. Sample a batch of math prompts from the **RLVR-only** pool.
2. Generate `G=8` independent sampled completions per prompt (configurable temperature and completion length).
3. Parse the **complete** generated answer and assign a deterministic reward: mathematically correct `1`, incorrect/malformed `0`.
4. Compare rational/decimal equivalents using exact arithmetic where possible; use SymPy safely for scalar algebra and verify linear-equation solutions. Never use Python `eval` or run model-generated code.
5. Compute group-relative advantages. **All-correct/all-incorrect groups contribute zero relative advantage**; handle zero variance without NaNs.
6. Apply a documented GRPO clipped policy loss using generated-token log-probabilities, proper completion/padding masks and an optional frozen-SFT-reference KL penalty. Make ratio construction, normalization and clipping mathematically explicit and unit-tested.
7. Log reward, mixed/all-correct/all-wrong group rates, entropy, KL, clip fraction, gradient norms, throughput and GPU usage.

Keep frozen reference inference and rollout sampling out of autograd; avoid retaining generation graphs. Prioritize a correct, reproducible baseline over novel optimizers.

### Controls

For every model size run:

| Arm | Initialization | Additional training |
|---|---|---|
| A | Random | SFT |
| B | Same SFT checkpoint | RLVR (GRPO) |
| C | Same SFT checkpoint | Continued SFT at an approximately matched additional compute budget |

Match data, seed policy, and evaluation wherever possible. Clearly report the FLOP estimate and limitations of compute matching. Run three independent seeds if feasible, otherwise at least one full seed and explicitly mark uncertainty.

## Evaluation

Keep the test sets fixed across all arms and sizes. Main decoding: greedy, max output length capped and reported. Optional sampled pass@k is secondary.

Report:

- Exact mathematical correctness, format-validity rate, output token count.
- Per-module accuracy for all five modules.
- In-distribution accuracy vs original/controlled extrapolation performance.
- Accuracy by operand digits and expression length; separate carry/borrow and division-specific difficulty if supported.
- RLVR gain (percentage points), continued-SFT gain, performance per FLOP/token, and uncertainty across seeds.
- RL diagnostic fraction of **mixed-outcome** groups, especially when the 1M model cannot discover correct rollouts or the larger model saturates.

Produce result CSV/JSON and figures (PNG + SVG): model scaling (log parameter axis); SFT/RLVR/continued-SFT bars; per-module breakdown; accuracy versus digit count; training/reward/entropy/KL curves; ID vs OOD; accuracy versus compute.

**Never fabricate missing results.** Distinguish findings from speculation, and report exclusions/data limitations.

## Suggested repository layout

```text
math-llm/
├── configs/
│   ├── config.yaml                    # Hydra defaults and global paths
│   ├── model/{1m,10m,100m}.yaml
│   ├── data/default.yaml
│   ├── trainer/{sft,continued_sft}.yaml
│   ├── rlvr/grpo.yaml
│   ├── eval/default.yaml
│   ├── tracking/wandb.yaml
│   └── experiment/{smoke,pilot,main,extended}.yaml
├── src/tinymath/
│   ├── data/{load.py,splits.py,char_tokenizer.py}
│   ├── model/gpt.py
│   ├── train/{sft.py,grpo.py,checkpoint.py}
│   ├── rewards/{parser.py,verify.py}
│   ├── eval/{evaluate.py,metrics.py}
│   ├── tracking/{wandb.py,provenance.py}
│   ├── config.py
│   └── plots/figures.py
├── scripts/{prepare_data.py,train_sft.py,train_rlvr.py,evaluate.py,run_all.py,plot.py}
├── tests/
├── data/                   # generated/downloaded; gitignored
├── runs/                   # checkpoints/local metrics; gitignored
├── outputs/                # Hydra run outputs; gitignored
├── wandb/                  # local W&B state; gitignored
├── .env                    # pre-existing W&B credentials; gitignored, never read into Git
├── .env.example            # optional key-free variable names only
├── .gitignore
├── pyproject.toml
├── uv.lock                 # committed dependency lock
└── README.md
```

Use `uv`-managed Python 3.11+ with PyTorch, Hugging Face Datasets, SymPy, Matplotlib, Hydra/OmegaConf, W&B and python-dotenv. Make every entry point work from `/workspace` inside the existing GPU Docker environment. Initialize W&B centrally in reusable tracking code for experiments, not with duplicated per-script login logic. Data preparation and annotation may have W&B provenance runs only as stages of a named experiment. Unit/integration tests and routine development checks must not connect to W&B or create offline runs.

## Milestones and acceptance checks

**M0 — Infrastructure**: initialize `uv` project, create and commit `uv.lock`, implement Hydra config composition/overrides and fixed workspace paths, confirm `torch.cuda.is_available()`, detect CUDA device, and implement seed/logging/checkpoint conventions and secure runtime credential loading. Validate tracking code with mocked W&B; verify real authentication/logging during the first actual experiment, not a separate infrastructure run. Never output secrets. Do not require Codex to modify the Docker host.

**M1 — Data/tokenizer**: download or stream, inspect, filter and split only five modules; round-trip char tokenizer; report lengths and verify no train/test overlap. Save split manifest and a few inspected samples.

**M2 — 1M SFT**: train a small overfit test then 2k and 20k pilot; verify masked loss decreases, saved checkpoint reloads, greedy accuracy and reward verifier work. Log performance and VRAM locally **and to W&B**, using Hydra overrides for experiment preset, model size and seed.

**M3 — RLVR smoke**: generate eight completions, score correctly, check zero-variance groups, verify gradients and one reproducible GRPO update; compare metrics before/after without claiming improvement from a single update. Record the smoke run and diagnostics in W&B.

**M4 — Complete 1M baseline**: run SFT, GRPO and continued SFT on the same evaluation splits; debug reward sparsity and sampling.

**M5 — Scale models**: repeat for 10M and 100M, keeping tokenizer/splits/budgets fixed. Adjust microbatch/accumulation only for memory, not effective budgets.

**M6 — OOD and reporting**: evaluate extrapolation, run any feasible extra seeds, produce plots/tables and a concise report discussing RLVR gains, sample efficiency, informative-group rate, and limitations.

## Implementation quality requirements

- Unit tests for arithmetic and symbolic answer verification, equivalent fractions/decimals, negative numbers, malformed completions, tokenizer round-trips, split leakage, target masks, group advantage handling and GRPO loss/gradients.
- Integration test: GPU or CPU fallback, one SFT step, checkpoint save/reload, one RLVR update, one evaluation batch.
- Seed Python, NumPy, PyTorch, dataset shuffling and generation. Save effective configs, environment versions, dataset revision and Git commit.
- Resume interrupted runs including their exact Hydra configuration and W&B run ID. Keep training metadata, W&B local state and generated outputs under the project workspace.
- State any unresolved data format/split inconsistencies explicitly rather than silently adapting the experiment.

## Commands expected from the implementation

Implement all script entry points with Hydra (`@hydra.main`) and document `uv run` commands using config group overrides. These are examples; config group names may be refined while preserving Hydra's compositional model:

```bash
cd /workspace
uv sync --locked
uv run python scripts/prepare_data.py experiment=smoke
uv run python scripts/train_sft.py model=1m trainer=sft experiment=pilot
uv run python scripts/train_rlvr.py model=1m rlvr=grpo experiment=smoke 'init_checkpoint=runs/1m/sft/best.pt'
uv run python scripts/evaluate.py model=1m eval=default 'checkpoint=runs/1m/rlvr/latest.pt'
uv run python scripts/run_all.py experiment=pilot
uv run python scripts/plot.py experiment=pilot
uv run pytest -q
```

Hydra CLI overrides must actually work in each command; make the default config composition and file paths valid. Include an example demonstrating W&B `project`, `group`, `name`, and `mode` overrides without putting any API key in command arguments. Document how to verify authenticated W&B logging and identify an offline/unsynced run if applicable. `uv.lock` and actual resolved Hydra configurations establish repeatable execution.

## Mandatory incremental local Git workflow

The implementation agent **must create small, validated local Git commits continuously**, rather than making one final commit. Git history is a deliverable and must provide a useful audit trail of the implementation. **Never push, fetch, pull, or otherwise contact a Git remote.** Do not request confirmation between routine implementation steps or commits.

### Initial repository inspection and preservation

1. Operate only within `/workspace`. Locate the existing Git repository root and inspect `git status --short`, `git branch --show-current`, `git log -5 --oneline` and the current `.gitignore` before changing anything.
2. If `/workspace` is not yet a Git repository, initialize a **local** repository there. Do not configure a remote. If it is already a repository, preserve its existing branch, commits, configuration and history.
3. Preserve pre-existing changes and staged files. Never silently incorporate or overwrite unrelated user work. Track which changes were created by the agent; use explicit staging paths to avoid including unrelated modifications. If a required file has overlapping user edits, avoid overwriting them and explain any resulting limitation.
4. Ensure `.gitignore` covers downloaded/processed datasets, checkpoints, model weights, generated run outputs, plot artifacts when large or regenerable, caches, temporary files, environment files and secrets. Keep source code, reproducible configurations, documentation and tests tracked. Do not use broad ignore rules that hide intended source files.

### Small-commit policy

- Use a commit for each coherent, reviewable unit: one feature, a related test suite, a fix, or a focused documentation/configuration update.
- Add tests in the same commit as the behavior they exercise whenever practical. Avoid committing known broken or syntactically invalid code. Partial milestones may be committed if the change is genuinely self-contained and validated.
- Run the relevant tests/validation before each commit and a broader suite at milestone boundaries. If a check is unavailable (e.g., no GPU or network), note that limitation explicitly in the final report and do not label the check as passed.
- Follow Conventional Commits with imperative, scoped messages, for example `feat(data): inspect and filter five mathematics modules`, `test(tokenizer): verify character round-trips`, `feat(sft): train with completion-only loss`, `fix(grpo): handle zero-variance groups`.
- Never create meaningless commits to achieve a quota, and never squash or amend existing commits. Do not reset, clean, rewrite history or force operations. Do not change branches unless explicitly instructed.
- Before committing, review `git diff --check`, the intended diff, `git diff --cached` and `git status --short`. Stage specific paths (`git add path/to/file ...`), not `git add -A` or `git add .`. Commit only the reviewed staged changes.
- Do not add remote URLs or invoke commands that communicate with remote Git hosts. Hugging Face dataset downloads used by the experiment are distinct from Git remote operations and are permitted within the stated project scope.

### Mandatory implementation loop

For **every** implementation unit, repeat without waiting for permission:

1. Select the smallest useful change in the current milestone.
2. Implement it and its appropriate tests.
3. Execute the smallest meaningful validation; fix errors before committing.
4. Review the working-tree diff and staged diff, including potential secrets, accidental data files and unrelated changes.
5. Stage only the agent-created files belonging to that unit.
6. Commit locally with a descriptive Conventional Commit message.
7. Record the new commit hash and move to the next unit.

At each milestone M0–M6, run milestone-appropriate integration checks and make sure no *agent-created source changes* remain uncommitted. Pre-existing user changes may remain and must not be modified merely to obtain a clean status. Keep experimental run outputs untracked unless they are small, deliberately versioned reproducibility artifacts.

### Suggested commit progression (adapt as implementation warrants)

```text
chore: initialize uv project, lock dependencies and ignore secrets
feat(config): compose Hydra configurations and experiment presets
feat(tracking): authenticate wandb from local environment securely
test(tracking): verify run metadata and resume behavior
feat(data): inspect and filter the five DeepMind modules
feat(data): build deterministic leakage-free pools
feat(tokenizer): add strict single-character vocabulary
feat(model): implement three GPT size configurations
feat(sft): implement completion-masked supervised training
feat(checkpoint): support resume and provenance capture
feat(reward): verify numeric and symbolic scalar answers
feat(grpo): implement grouped sampling and advantages
feat(grpo): implement masked clipped policy objective
feat(eval): add module and OOD metrics
feat(experiments): orchestrate pilot and full runs
test(integration): cover SFT-to-GRPO smoke flow
docs: describe validated commands and research protocol
```

This sequence is illustrative, not a strict linear dependency graph. Prefer extra small meaningful commits when necessary; do not commit an untested large milestone as one change.

### Reporting and acceptance of Git work

The final response must include the actual commands/tests executed and their outcomes, completed milestones, limitations, and the output of `git log --oneline` (or an accurate selected range with commit hashes) and `git status --short`. Include W&B tracking/sync status only when experiments were run or diagnosed, or the user explicitly requests it; omit it for routine development and documentation work. Explicitly confirm that commits are **local only** and no pushes occurred. Never fabricate commit IDs, tests, experimental results or completed milestones. A commit is not evidence that its feature is correct: executed checks remain required.

---

## Instruction to coding agent

Implement this plan autonomously and incrementally in the existing repository, obeying the mandatory Git workflow. **Use uv for environment and dependency management, Hydra for all experiment configuration, and W&B for every experiment/evaluation; securely load its API key from the existing local `.env` without exposing it.** **Do not merely scaffold files or defer functional validation:** execute smoke tests, then deliver working SFT, mathematical verifier and GRPO pipelines, and document the checks actually executed. Start with M0–M3, then continue into the remaining feasible milestones while keeping the full 1M/10M/100M experiment configurable and resumable. Make multiple small **local** commits throughout implementation; never push to a remote. Preserve the strict five-module scope and single-character tokenization. Do not change Docker host configuration or access files outside `/workspace`. Prioritize correctness, reproducibility, incremental tests and reviewable Git history over speed.
