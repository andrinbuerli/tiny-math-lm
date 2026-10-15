"""W&B-tracked controlled optimization, resume, and real greedy v3 evaluation.

Scripted training trajectories deliberately test plumbing rather than on-policy
learning. Both the optimizer and the final greedy evaluation use a real tiny GPT.
Invoke twice with identical options to exercise persisted W&B/checkpoint resume.
"""

from __future__ import annotations

import json
import logging
import string
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import hydra
import torch
from omegaconf import DictConfig
from torch.nn.attention import SDPBackend, sdpa_kernel

from tinymath.checkpoint import compatibility_metadata, save_checkpoint, sha256_file
from tinymath.config import workspace_path
from tinymath.data.char_tokenizer import CharacterTokenizer
from tinymath.data.load import SELECTED_MODULES
from tinymath.data.prompts import format_prompt, format_reasoning_display
from tinymath.data.reasoning import annotate_row
from tinymath.eval.evaluate import evaluate_checkpoint
from tinymath.generation import RolloutConfig, generate_rollouts
from tinymath.model.gpt import GPTConfig, TinyMathGPT
from tinymath.rewards.calls import scalar_text
from tinymath.tracking.provenance import read_wandb_run_id
from tinymath.tracking.wandb import finish_wandb, initialize_tracked_run
from tinymath.train import grpo_runner

LOGGER = logging.getLogger(__name__)


class ControlledInterruption(Exception):
    pass


class ScriptedPolicy(torch.nn.Module):
    def __init__(self, tokenizer, texts, config):
        super().__init__()
        self.config = config
        self.scripts = [tokenizer.encode(text, add_eos=True) for text in texts]
        self.tokenizer = tokenizer

    def forward_with_cache(self, ids, cache):
        position = cache or 0
        logits = torch.full((*ids.shape, self.tokenizer.vocab_size), -100., device=ids.device)
        for lane, script in enumerate(self.scripts):
            token = script[position] if position < len(script) else self.tokenizer.eos_id
            logits[lane, -1, token] = 100.
        return logits, position + 1


def prepare_fixture(root: Path, config: DictConfig):
    """Small synthetic disjoint parent pools across exactly the five modules."""
    bundle = root / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    tokenizer = CharacterTokenizer.from_texts([string.printable], protocol="reasoning_v3")
    tokenizer.save(bundle / "char_tokenizer.json")
    rows_by_prompt = {}
    for split, a in (("sft_train", 10), ("sft_validation", 20), ("rlvr", 30), ("test", 40)):
        sources = [
            (f"Calculate {a} + 2.", str(a + 2), "arithmetic__add_or_sub"),
            (f"Calculate {a} * 2.", str(a * 2), "arithmetic__mul"),
            (f"Calculate {a} / 2.", str(a // 2), "arithmetic__div"),
            (f"Calculate ({a} + 1)*2.", str((a + 1) * 2), "arithmetic__mixed"),
            (f"Solve 2*x + {a} = {a + 6} for x.", "3", "algebra__linear_1d"),
        ]
        rows = []
        for question, answer, module in sources:
            row = annotate_row({"question": question, "answer": answer, "module": module,
                                "category": "synthetic-validation"}, protocol="reasoning_v3")
            row["is_sub_task"] = False
            rows.append(row)
            rows_by_prompt[format_prompt(question, protocol="reasoning_v3")] = row
        (bundle / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "dataset_revision": "synthetic-controlled-validation-v1",
        "selected_modules": list(config.data.modules),
        "context_length": int(config.model.context_length),
        "tokenizer": {"sha256": tokenizer.sha256},
        "annotation": {"protocol": "reasoning_v3", "annotation_sha256": "synthetic-v3-v1"},
        "split_manifest_sha256": sha256_file(bundle / "rlvr.jsonl"),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    initial_path = root / "controlled_initial.pt"
    if not initial_path.exists():
        model_config = GPTConfig.from_mapping({**dict(config.model), "vocab_size": tokenizer.vocab_size})
        model = TinyMathGPT(model_config)
        save_checkpoint(initial_path, {
            "model": model.state_dict(), "model_config": model_config.__dict__,
            "tokenizer_sha256": tokenizer.sha256, "step": 0,
            "training_arm": "synthetic_random_initialization",
            "compatibility": compatibility_metadata(
                protocol="reasoning_v3", annotation_sha256="synthetic-v3-v1",
                tokenizer_sha256=tokenizer.sha256, bundle_sha256=sha256_file(bundle / "manifest.json"),
                context_length=int(config.model.context_length),
            ),
        })
    return bundle, initial_path, rows_by_prompt


def controlled_sampler(rows):
    def sample(model, prompt_ids, *, tokenizer, rollout, group_size, **kwargs):
        texts = []
        for ids in prompt_ids:
            row = rows[tokenizer.decode(ids.tolist())]
            for lane in range(group_size):
                answer = row["answer"] if lane % 3 == 0 else scalar_text(Fraction(row["answer"]) + 1)
                if lane == 2:
                    texts.append(format_reasoning_display(
                        "step:auxiliary arithmetic\ncall:add(999,999) -> 1998", row["answer"]
                    ))
                else:
                    texts.append(format_reasoning_display(row["scratchpad"], answer) if lane % 3 != 2 else "malformed")
        result = generate_rollouts(
            ScriptedPolicy(tokenizer, texts, model.config), prompt_ids, group_size=group_size,
            tokenizer=tokenizer, rollout=rollout, **kwargs,
        )
        with torch.no_grad():
            old_logs = grpo_runner.completion_log_probs(
                model, prompt_ids, result.completion_ids, temperature=kwargs["temperature"]
            )
        return replace(result, old_log_probs=torch.where(result.loss_mask, old_logs, 0.).detach())
    return sample


@hydra.main(version_base="1.3", config_path="../configs", config_name="validate_reasoning_v3")
def main(config: DictConfig) -> None:
    if set(config.data.modules) != set(SELECTED_MODULES):
        raise ValueError("Validation is restricted to the five selected modules")
    if not config.validation.controlled_trajectories:
        raise ValueError("This entry point is exclusively a controlled plumbing check")
    root = workspace_path(str(config.validation.root))
    run_dir = root / "grpo"
    run = initialize_tracked_run(config, run_dir, job_type="smoke-optimization")
    exit_code = 0
    try:
        bundle, initial, rows = prepare_fixture(root, config)
        run.config.update({
            "validation_kind": "scripted-off-policy-plumbing-only",
            "initialization": "synthetic random tiny GPT; not a trained SFT model",
            "dataset_revision": "synthetic-controlled-validation-v1",
            "bundle_sha256": sha256_file(bundle / "manifest.json"),
            "tokenizer_sha256": CharacterTokenizer.load(bundle / "char_tokenizer.json").sha256,
            "scripted_entropy": True,
            "validation_includes_forbidden_calls": True,
        }, allow_val_change=True)
        original_save = grpo_runner.save_checkpoint
        marker = root / "interruption_record.json"
        def interrupt_after_save(path, payload):
            original_save(path, payload)
            if config.validation.interrupt_once and not marker.exists() and path.name == "latest_inference.pt":
                marker.write_text(json.dumps({"wandb_run_id": str(run.id), "update": payload["update"]}))
                raise ControlledInterruption("Checkpoint saved; rerun this exact command to resume")
        with sdpa_kernel(SDPBackend.MATH), patch.object(grpo_runner, "sample_group_rollouts", controlled_sampler(rows)), patch.object(grpo_runner, "save_checkpoint", interrupt_after_save):
            result = grpo_runner.train_grpo(config=config, bundle_dir=bundle, init_checkpoint=initial,
                                            run_dir=run_dir, run=run)
        if marker.exists():
            assert json.loads(marker.read_text())["wandb_run_id"] == str(run.id)
        checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        before = torch.load(initial, map_location="cpu", weights_only=False)
        changed = any(not torch.equal(value, checkpoint["model"][name]) for name, value in before["model"].items())
        assert changed, "Controlled mixed rewards must exercise an optimizer update"
        run.summary.update({"validation/optimizer_changed_weights": changed, "validation/resumed_same_run": marker.exists(),
                            "validation/controlled_reward": result["final_reward"]})
        (root / "optimization_result.json").write_text(json.dumps(result, indent=2) + "\n")
    except ControlledInterruption as exc:
        exit_code = 1
        LOGGER.info("Controlled interruption: %s", exc)
        return
    except BaseException:
        exit_code = 1
        raise
    finally:
        finish_wandb(run, exit_code=exit_code)

    eval_run = initialize_tracked_run(config, root / "evaluation", job_type="evaluation")
    exit_code = 0
    try:
        summary = evaluate_checkpoint(
            checkpoint_path=Path(result["inference_checkpoint"]), bundle_dir=bundle,
            output_dir=root / "evaluation", split_names=list(config.eval.splits), max_new_tokens=int(config.eval.max_new_tokens),
            batch_size=int(config.eval.batch_size), rollout=RolloutConfig.from_config(config.rollout),
        )
        # Greedy predictions are real model outputs, not the scripted training trajectories.
        eval_run.config.update({"validation_kind": "real-greedy-tiny-model",
            "source_training_wandb_run_id": read_wandb_run_id(run_dir),
            "checkpoint_sha256": summary["checkpoint_sha256"], "bundle_sha256": summary["bundle_sha256"],
            "rollout": summary["rollout"]}, allow_val_change=True)
        eval_run.log({f"eval/test/{key}": value for key, value in summary["splits"]["test"].items()
                      if isinstance(value, (int, float))})
        LOGGER.info("Real greedy evaluation: %s", summary["splits"]["test"])
    except BaseException:
        exit_code = 1
        raise
    finally:
        finish_wandb(eval_run, exit_code=exit_code)


if __name__ == "__main__":
    main()
