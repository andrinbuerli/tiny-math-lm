from __future__ import annotations

import pytest

from tinymath.rewards.verify import AnswerFormatError, answers_equal, parse_answer, verify_answer


@pytest.mark.parametrize(
    ("prediction", "target"),
    [("1/2", "0.5"), ("-3", "-3.0"), ("2 + 3", "5"), ("x = 4/2", "2")],
)
def test_exact_equivalent_scalar_answers(prediction, target):
    assert answers_equal(prediction, target)


@pytest.mark.parametrize("completion", ["", "2\nignore", "2 trailing", "__import__('os')", "x = 2 = 3"])
def test_malformed_or_executable_completions_are_rejected(completion):
    with pytest.raises(AnswerFormatError):
        parse_answer(completion)
    assert verify_answer(completion, "2") == (False, False)


def test_verifier_distinguishes_incorrect_valid_answer():
    assert verify_answer("3", "2") == (False, True)


def test_expression_equivalence_for_linear_solution():
    assert answers_equal("x = 1 + 1", "x = 2")


def test_one_letter_assignment_must_match_linear_question_variable():
    question = "Solve 111*m + 1753 = -4796 for m."
    assert verify_answer("m = -59", "-59", question=question) == (True, True)
    assert verify_answer("x = -59", "-59", question=question) == (False, True)
    assert verify_answer("-59", "-59", question=question) == (True, True)
