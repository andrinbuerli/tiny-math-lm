import torch

from tinymath.model.hf_qwen import completion_loss


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
