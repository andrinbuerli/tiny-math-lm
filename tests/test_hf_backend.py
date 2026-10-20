import pytest
import torch
from omegaconf import OmegaConf

from tinymath.model.hf_qwen import completion_loss
from tinymath.runtime_gpu import gpu_capacity_available
from tinymath.train import hf_sft


class TinyLogitModel(torch.nn.Module):
    def __init__(self, vocab_size=7):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, attention_mask, use_cache=False):
        batch, length = input_ids.shape
        return type("Output", (), {"logits": self.logits.expand(batch, length, -1)})()


def test_completion_loss_counts_only_completion_targets_including_native_eos():
    model = TinyLogitModel()
    input_ids = torch.tensor([[1, 2, 3, 6, 6]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, 4, -100]])

    loss_sum, token_count = completion_loss(model, input_ids, attention_mask, labels)

    assert token_count == 1
    loss_sum.backward()
    assert model.logits.grad is not None


def test_completion_loss_preserves_eos_when_pad_uses_same_token_id():
    model = TinyLogitModel()
    eos_id = 6
    input_ids = torch.tensor([[1, 2, 3, eos_id, eos_id]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    labels = torch.tensor([[-100, -100, -100, eos_id, -100]])

    _loss_sum, token_count = completion_loss(model, input_ids, attention_mask, labels)

    assert token_count == 1


def test_qwen_training_admission_requires_utilization_and_vram_headroom():
    config = OmegaConf.create(
        {
            "runtime": {
                "parallel_gpu_max_utilization": 75,
                "parallel_gpu_min_free_vram_gb": 2,
                "parallel_model_vram_gb": {"qwen3_0_6b_base": 10},
            }
        }
    )
    assert gpu_capacity_available(config, "qwen3_0_6b_base", 60, 12 * 1024)
    assert not gpu_capacity_available(config, "qwen3_0_6b_base", 76, 16 * 1024)
    assert not gpu_capacity_available(config, "qwen3_0_6b_base", 60, 12 * 1024 - 1)


class TinyCachedSplit:
    def __init__(self, _cache_dir, split):
        self.split = split

    def __len__(self):
        return 4

    def batch(self, indices, *, pad_id, device):
        del pad_id
        input_ids = torch.tensor([[1, 2, 3, 6] for _ in indices], device=device)
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, 3, 6] for _ in indices], device=device)
        return input_ids, attention, labels


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(7))

    def forward(self, input_ids, attention_mask, use_cache=False):
        batch, length = input_ids.shape
        return type("Output", (), {"logits": self.logits.expand(batch, length, -1)})()

    def save_pretrained(self, path, safe_serialization=True):
        del safe_serialization
        path.mkdir(parents=True, exist_ok=True)
        (path / "tiny-model.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"tiny fixture weights")


class TinyTokenizer:
    def save_pretrained(self, path):
        (path / "tiny-tokenizer.json").write_text("{}", encoding="utf-8")


def test_cpu_fixture_resumes_deterministically_from_optimizer_boundary(monkeypatch, tmp_path):
    config = OmegaConf.create(
        {
            "seed": 18,
            "experiment": {"phase": "pilot", "pilot_train_rows": 4, "pilot_validation_rows": 2},
            "model": {"revision": "revision", "context_length": 512, "gradient_checkpointing": True},
            "trainer": {
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "batch_size": 1,
                "gradient_accumulation": 2,
                "max_epochs": 1,
                "max_steps": 2,
                "eval_every": 100,
                "save_every": 1,
                "warmup_ratio": 0.0,
                "grad_clip": 1.0,
                "early_stopping_patience_epochs": 1,
            },
            "eval": {"batch_size": 1},
        }
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = {
        "tokenizer_identity": {"pad_token_id": 6},
        "tokenizer_sha256": "tokenizer-hash",
        "model_config_sha256": "config-hash",
    }
    monkeypatch.setattr(hf_sft, "_find_cache", lambda _config: (cache_dir, manifest))
    monkeypatch.setattr(hf_sft, "HFTokenSplit", TinyCachedSplit)
    monkeypatch.setattr(hf_sft, "load_model", lambda *_args, **_kwargs: TinyTrainModel())
    monkeypatch.setattr(hf_sft, "load_tokenizer", lambda _config: TinyTokenizer())
    checkpoint_dir = tmp_path / "run"

    def stop_after_first_boundary(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        if path.name == "latest.pt" and payload["step"] == 1:
            raise RuntimeError("simulated interruption after complete optimizer update")

    monkeypatch.setattr(hf_sft, "save_checkpoint", stop_after_first_boundary)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        hf_sft.train_hf_sft(
            config=config,
            run_dir=checkpoint_dir,
            run=None,
            device_override=torch.device("cpu"),
        )
    interrupted = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert interrupted["step"] == 1
    assert interrupted["cursor"] == 2

    def save_checkpoint(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    monkeypatch.setattr(hf_sft, "save_checkpoint", save_checkpoint)
    result = hf_sft.train_hf_sft(
        config=config,
        run_dir=checkpoint_dir,
        run=None,
        device_override=torch.device("cpu"),
    )
    assert result["step"] == 2
    assert result["examples_seen"] == 4
    assert result["tokens_seen"] > 0
    assert (checkpoint_dir / "best" / "experiment_identity.json").is_file()
    completed = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert completed["step"] == 2
    assert completed["epoch"] == 1
