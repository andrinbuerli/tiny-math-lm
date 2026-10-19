from tinymath.generation import RolloutConfig
from tinymath.train.grpo_runner import _prompt_buckets


def test_left_padding_is_default_and_old_rollout_config_stays_legacy():
    assert RolloutConfig().sampler_version == 2
    assert RolloutConfig().prompt_batching == "left_pad"
    legacy = RolloutConfig.from_config({"rollout_version": 3})
    assert legacy.sampler_version == 1
    assert legacy.prompt_batching == "exact_length"


def test_left_padding_buckets_by_effective_completion_cap():
    lengths = [19, 41, 17, 27, 25, 16, 52, 22, 500, 500, 504]
    buckets = _prompt_buckets(
        lengths, max_completion_tokens=128, context_length=512, strategy="left_pad"
    )
    assert buckets == [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9], [10]]


def test_exact_length_strategy_remains_available_for_legacy_replay():
    lengths = [8, 5, 8, 5, 12]
    assert _prompt_buckets(
        lengths, max_completion_tokens=4, context_length=16, strategy="exact_length"
    ) == [[0, 2], [1, 3], [4]]
