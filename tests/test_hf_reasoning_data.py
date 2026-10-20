import pytest

from tinymath.data.hf_reasoning import (
    encode_completion,
    identity_hash,
    render_completion,
    render_prompt,
    select_parent_rows,
    select_training_rows,
)


class FakeTokenizer:
    eos_token_id = 2
    unk_token_id = 0

    def __init__(self):
        self.ids = {"<think>": 3, "</think>": 4}

    def convert_tokens_to_ids(self, text):
        return self.ids.get(text, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        if text in self.ids:
            return [self.ids[text]]
        return [10 + ord(char) for char in text]


def _row(**kwargs):
    return {
        "module": "arithmetic__add_or_sub",
        "question": "Calculate <scp> x.",
        "answer": "42",
        "scratchpad": "step:add:20+22=42",
        "is_sub_task": False,
        **kwargs,
    }


def test_native_rendering_rewrites_only_structural_scratchpad_markers():
    assert render_prompt("Calculate <scp> x.") == "Q: Calculate <scp> x.\n"
    assert render_completion("step:add:20+22=42", "42") == (
        "<think>\nstep:add:20+22=42\n</think>\nA: 42"
    )


def test_training_pool_keeps_parents_and_only_eligible_nonoverlapping_subtasks():
    parent = _row()
    eligible = _row(is_sub_task=True, training_eligible=True, overlapping_splits=[])
    ineligible = _row(is_sub_task=True, training_eligible=False, overlapping_splits=["test"])
    assert select_training_rows([parent, eligible, ineligible]) == [parent, eligible]
    assert select_parent_rows([parent, eligible]) == [parent]


def test_eligible_subtask_with_cross_pool_overlap_fails_closed():
    with pytest.raises(ValueError, match="cross-pool overlap"):
        select_training_rows([_row(is_sub_task=True, training_eligible=True,
                                   overlapping_splits=["rlvr"])])


def test_completion_labels_include_native_eos_but_mask_prompt():
    ids, labels, prompt_length = encode_completion(FakeTokenizer(), _row())
    assert prompt_length > 0
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length:] == ids[prompt_length:]
    assert labels[-1] == FakeTokenizer.eos_token_id
    assert len(ids) == len(labels)


def test_true_eos_is_not_confused_with_padding_and_over_context_rows_are_rejected():
    ids, labels, _ = encode_completion(FakeTokenizer(), _row())
    padded_labels = labels + [-100, -100]
    assert padded_labels[-3] == FakeTokenizer.eos_token_id
    with pytest.raises(ValueError, match="refusing to truncate or drop"):
        encode_completion(FakeTokenizer(), _row(), context_length=40, completion_cap=512)


def test_identity_hash_ignores_input_order_but_tracks_module_question_target():
    first = _row()
    second = _row(question="Calculate 2 + 2.")
    assert identity_hash([first, second]) == identity_hash([second, first])
    assert identity_hash([first])[0] != identity_hash([second])[0]
