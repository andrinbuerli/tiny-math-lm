from __future__ import annotations

import random
import re
from copy import deepcopy
from fractions import Fraction

import pytest

from tinymath.data.reasoning import (
    _Builder,
    _decimal,
    _ExpressionParser,
    _fraction,
    _source_expression,
    annotate_row,
    default_sequence_length,
    generate_subtask_rows,
    reduce_to_budget,
    verify_annotation,
)


def row(question: str, answer: str, module: str) -> dict[str, str]:
    return {"question": question, "answer": answer, "category": "train-easy", "module": module}


@pytest.mark.parametrize(
    ("source", "answer", "module"),
    [
        ("What is -32 + -13684093925?", "-13684093957", "arithmetic__add_or_sub"),
        ("What is the product of -32.8 and 199809?", "-6553735.2", "arithmetic__mul"),
        ("What is -274 divided by 2770?", "-137/1385", "arithmetic__div"),
        ("Evaluate (7/(-7) - 4) + 6 + 3", "4", "arithmetic__mixed"),
        ("Solve -3603 + 4878 = -75*d for d.", "-17", "algebra__linear_1d"),
    ],
)
def test_golden_module_rows_derive_exact_answer(source: str, answer: str, module: str) -> None:
    source_row = row(source, answer, module)
    original = deepcopy(source_row)
    annotated = annotate_row(source_row)
    assert source_row == original
    assert {key: annotated[key] for key in source_row} == source_row
    assert annotated["scratchpad"]
    assert "<scp>" not in annotated["scratchpad"]
    assert "A:" not in annotated["scratchpad"]
    assert annotated["retained_length"] <= 512


def test_decimal_arithmetic_and_negative_subtraction_are_exact() -> None:
    decimal_row = row("-6.14+0.4764252833", "-5.6635747167", "arithmetic__add_or_sub")
    decimal_annotation = annotate_row(decimal_row)
    verify_annotation(decimal_row, decimal_annotation)
    result = decimal_annotation
    assert "scale:10=>-5.6635747167" in result["scratchpad"]
    subtraction_row = row("Subtract -324470729 from 11.422.", "324470740.422", "arithmetic__add_or_sub")
    result = annotate_row(subtraction_row)
    verify_annotation(subtraction_row, result)
    assert result["scratchpad"]
    assert _fraction("-5.6635747167") == Fraction(-56635747167, 10**10)
    assert _decimal(Fraction(1, 2)) == "0.5"


@pytest.mark.parametrize(
    ("question", "answer", "module"),
    [
        ("999 + 1", "1000", "arithmetic__add_or_sub"),
        ("1000 - 1", "999", "arithmetic__add_or_sub"),
        ("-6.14+0.4764252833", "-5.6635747167", "arithmetic__add_or_sub"),
        ("123 * 45", "5535", "arithmetic__mul"),
        ("What is the product of -32.8 and 199809?", "-6553735.2", "arithmetic__mul"),
        ("What is 1 divided by 20?", "1/20", "arithmetic__div"),
        ("What is -274 divided by 2770?", "-137/1385", "arithmetic__div"),
        ("Evaluate (8 + 4) * 3", "36", "arithmetic__mixed"),
        ("Evaluate 2 + 3 + 0", "5", "arithmetic__mixed"),
        ("Solve 3*x + 2 = 11 for x.", "3", "algebra__linear_1d"),
    ],
)
def test_reasoning_v2_is_forward_ordered_and_answer_is_separate(
    question: str, answer: str, module: str
) -> None:
    source = row(question, answer, module)
    annotated = annotate_row(source, protocol="reasoning_v2")
    trace = str(annotated["scratchpad"])
    assert annotated["trace_version"] == "reasoning_v2"
    assert trace.startswith("step:")
    assert "A:" not in trace
    assert not re.search(r"(?m)^t\d+=.*=", trace)
    assert all(
        not line.startswith("=>") or _fraction(line[2:]) != _fraction(answer)
        for line in trace.splitlines()
    )
    verify_annotation(source, annotated)
    with pytest.raises(ValueError, match="result-first"):
        verify_annotation(source, {**annotated, "scratchpad": "t0=+(1,2)=3\n1,2,0:3,0"})
    with pytest.raises(ValueError, match="result-first"):
        verify_annotation(source, {**annotated, "scratchpad": trace.rsplit("\n", 1)[0]})


def test_reasoning_v2_keeps_full_forward_derivation_when_within_budget():
    source = row("12345 * 6789", "83810205", "arithmetic__mul")
    annotated = annotate_row(source, protocol="reasoning_v2")
    assert not annotated["collapsed_categories"]
    assert annotated["retained_length"] <= 512
    assert annotated["scratchpad"].endswith("9,4,0:3,1\n0,7,1:8,0")
    assert "=>9740205" in annotated["scratchpad"]
    assert "83810205" not in annotated["scratchpad"]
    verify_annotation(source, annotated)


def test_reasoning_v2_verifier_replays_nondefault_trace_budget():
    source = row("12345 * 6789", "83810205", "arithmetic__mul")
    annotation = annotate_row(source, max_tokens=10**9, protocol="reasoning_v2")
    assert annotation["annotation_max_tokens"] == 10**9
    verify_annotation(source, annotation)


def test_carries_borrows_and_arbitrary_operation_indices() -> None:
    carry_row = row("999 + 1", "1000", "arithmetic__add_or_sub")
    carry_annotation = annotate_row(carry_row)
    verify_annotation(carry_row, carry_annotation)
    carry = carry_annotation["scratchpad"]
    assert "9,1,0:0,1" in carry and "carry:1" in carry
    borrow_row = row("1000 - 1", "999", "arithmetic__add_or_sub")
    borrow_annotation = annotate_row(borrow_row)
    verify_annotation(borrow_row, borrow_annotation)
    borrow = borrow_annotation["scratchpad"]
    assert "0,1,0:9,1" in borrow
    nested = row(" + ".join(["1"] * 12), "12", "arithmetic__mixed")
    trace = annotate_row(nested)["scratchpad"]
    assert "t10=" in trace


def test_seeded_random_signed_addition_and_subtraction_transitions() -> None:
    rng = random.Random(11018)
    for _ in range(40):
        left = rng.randint(-999_999, 999_999)
        right = rng.randint(-999_999, 999_999)
        operation = rng.choice(("+", "-"))
        answer = left + right if operation == "+" else left - right
        result = annotate_row(
            row(
                f"What is {left} {operation} {right}?",
                str(answer),
                "arithmetic__add_or_sub",
            )
        )
        assert result["retained_length"] <= 512
        assert result["scratchpad"]


def test_multiplication_has_partial_products_zero_rows_and_golden_budget_reduction() -> None:
    source = row("12345 * 6789", "83810205", "arithmetic__mul")
    annotated = annotate_row(source)
    verify_annotation(source, annotated)
    assert "p0=digit:9,8,7,6" in annotated["scratchpad"]
    assert "M6789@0=111105" in annotated["scratchpad"]
    assert "M6789@1=987600" in annotated["scratchpad"]
    assert "M6789@2=8641500" in annotated["scratchpad"]
    assert "M6789@3=74070000" in annotated["scratchpad"]
    assert "t2=t1+p3=83810205" in annotated["scratchpad"]
    assert "t2=t1+p3 ->" not in annotated["scratchpad"]
    assert "t1=" in annotated["scratchpad"] and "p3=" in annotated["scratchpad"]
    definitions = set(re.findall(r"(?m)^(?:t|p)\d+(?==)", annotated["scratchpad"]))
    references = set(re.findall(r"\b(?:t|p)\d+\b", annotated["scratchpad"]))
    assert references <= definitions
    assert annotated["collapsed_categories"] == {"addition": 1}
    assert (annotated["full_length"], annotated["retained_length"]) == (554, 480)
    zero = annotate_row(row("What is the product of 0 and 1559?", "0", "arithmetic__mul"))
    assert "M1559@0=0" in zero["scratchpad"]


def test_long_division_keeps_quotient_zero_and_reduces_fraction_exactly() -> None:
    source = row("What is 1 divided by 20?", "1/20", "arithmetic__div")
    annotation = annotate_row(source)
    verify_annotation(source, annotation)
    trace = annotation["scratchpad"]
    assert ":0," in trace
    assert "gcd=1;1/20" in trace
    source_trace = annotate_row(row("What is 12 divided by 3?", "4", "arithmetic__div"))["scratchpad"]
    assert "12:4,0" in source_trace
    assert source_trace.endswith("gcd=3;4/1")


def test_mixed_expression_postorder_and_nested_parentheses() -> None:
    source = row("Calculate (47 - (-83 - -142))*(-1)/(2/(-4)).", "-24", "arithmetic__mixed")
    annotation = annotate_row(source)
    verify_annotation(source, annotation)
    trace = annotation["scratchpad"]
    ids = [line.split("=", 1)[0] for line in trace.splitlines() if line.startswith("t")]
    assert ids == [f"t{i}" for i in range(len(ids))]
    assert "t" in trace and "=" in trace


def test_mixed_nonterminating_rational_intermediate_uses_exact_equation() -> None:
    source = row("Calculate 2869/29 + -99.", "-2/29", "arithmetic__mixed")
    annotation = annotate_row(source)
    verify_annotation(source, annotation)
    trace = annotation["scratchpad"]
    assert "t0=/(2869,29)=2869/29" in trace
    assert "t1=+(t0,-99)=-2/29" in trace
    assert trace.splitlines()[-1] == "t1=+(t0,-99)=-2/29"


def test_reduced_rational_helper_references_are_verified_exactly() -> None:
    source = row(
        "Calculate (5/(-80)*-244 + (-28)/(-16))/((-9)/(-36)).",
        "68",
        "arithmetic__mixed",
    )
    annotation = annotate_row(source)
    verify_annotation(source, annotation)
    assert "t5=t3+t4=17" in annotation["scratchpad"]


def test_repeated_multiplication_nodes_have_unique_references_and_fit_budget() -> None:
    source = row("(-6)/4*((-9)/(-8) - 1)*149804/(-6609)", "17/4", "arithmetic__mixed")
    annotation = annotate_row(source)
    verify_annotation(source, annotation)
    trace = annotation["scratchpad"]
    identifiers = [line.split("=", 1)[0] for line in trace.splitlines() if line[:1] in {"p", "t"}]
    assert len(identifiers) == len(set(identifiers))
    assert "p8=" in trace and "t12=" in trace
    assert annotate_row(source)["retained_length"] <= 512


def test_linear_equation_collects_and_substitutes_exactly() -> None:
    source = row("Solve -74*d + 5537428 = 5540240 for d.", "-38", "algebra__linear_1d")
    annotation = annotate_row(source)
    verify_annotation(source, annotation)
    trace = annotation["scratchpad"]
    assert "move:" in trace
    assert "verify:" in trace
    assert trace.endswith("verify:5540240=5540240")


def test_reasoning_v2_linear_trace_ends_with_solution_without_generic_verification():
    source = row("Solve -21*o + 32 + 120 = -37 for o.", "9", "algebra__linear_1d")
    annotation = annotate_row(source, protocol="reasoning_v2")
    trace = str(annotation["scratchpad"])
    assert "collect:" in trace
    assert "move:-21o=-189" in trace
    assert "divide both sides by -21" in trace
    assert "verify:substitution satisfies the original equation" not in trace
    assert trace.endswith("o=9")
    assert "-189" in trace
    verify_annotation(source, annotation)


@pytest.mark.parametrize(
    ("question", "answer", "module", "call"),
    [
        ("Calculate 25 + 32.", "57", "arithmetic__add_or_sub", "call:add(25,32) -> 57"),
        ("Calculate 25 - 32.", "-7", "arithmetic__add_or_sub", "call:subtract(25,32) -> -7"),
        ("Calculate -7 * 8.", "-56", "arithmetic__mul", "call:multiply(-7,8) -> -56"),
        ("Calculate -189 / -21.", "9", "arithmetic__div", "call:divide(-189,-21) -> 9"),
        ("Calculate 2869/29 + -99.", "-2/29", "arithmetic__mixed", "call:add(2869/29,-99) -> -2/29"),
    ],
)
def test_reasoning_v3_solves_atomic_tasks_without_delegating(question, answer, module, call):
    source = row(question, answer, module)
    if module == "arithmetic__mixed":
        # This source contains a real division subexpression, but never delegates its root addition.
        annotation = annotate_row(source, protocol="reasoning_v3")
        assert call not in annotation["scratchpad"]
    else:
        annotation = annotate_row(source, protocol="reasoning_v3")
        assert "call:" not in annotation["scratchpad"]
        assert annotation["tool_calls_allowed"] is False
        assert generate_subtask_rows(source, annotation, source_split="sft_train", parent_row_index=0) == []
    verify_annotation(source, annotation)


def test_reasoning_v3_nested_calls_are_postorder_and_fraction_tasks_are_parenthesized():
    source = row(
        "Calculate (47 - (-83 - -142))*(-1)/(2/(-4)).", "-24", "arithmetic__mixed"
    )
    annotation = annotate_row(source, protocol="reasoning_v3")
    calls = [line for line in annotation["scratchpad"].splitlines() if line.startswith("call:")]
    assert calls == [
        "call:subtract(-83,-142) -> 59",
        "call:subtract(47,59) -> -12",
        "call:multiply(-12,-1) -> 12",
        "call:divide(2,-4) -> -1/2",
    ]
    subtasks = generate_subtask_rows(source, annotation, source_split="sft_train", parent_row_index=0)
    assert subtasks[-1]["question"] == "Calculate 2 / -4."
    assert subtasks[-1]["answer"] == "-1/2"
    assert subtasks[-1]["module"] == "arithmetic__div"
    assert subtasks[-1]["subtask_operation_count"] == 1
    assert subtasks[-1]["subtask_schema_version"] == 2
    # Re-annotation consumes the structured operands, not the slashes in their text.
    trace = annotate_row(subtasks[-1], protocol="reasoning_v3")["scratchpad"]
    assert "call:" not in trace
    assert "/(12,-1/2)" in annotation["scratchpad"]


def test_fraction_subtraction_subtask_is_one_operator_and_matches_reported_example():
    source = row("Calculate (-1.06 - -4.10201)*2.", "6.08402", "arithmetic__mixed")
    annotation = annotate_row(source, protocol="reasoning_v3")
    subtasks = generate_subtask_rows(source, annotation, source_split="sft_train", parent_row_index=0)
    assert len(subtasks) == 1
    task = subtasks[0]
    assert task["question"] == "Calculate (-53/50) - (-410201/100000)."
    assert task["module"] == "arithmetic__add_or_sub"
    assert "/(" not in task["scratchpad"]  # No spurious fraction-construction divisions.
    trace = annotate_row(task, protocol="reasoning_v3")["scratchpad"]
    assert "call:" not in trace
    assert task["tool_calls_allowed"] is False
    verify_annotation(task, task)


@pytest.mark.parametrize(("expression", "answer"), [
    ("(1/3)+(1/7)", "10/21"), ("(1/3)-(1/7)", "4/21"),
    ("(1/3)*(1/7)", "1/21"), ("(1/3)/(1/7)", "7/3"),
    ("(-1/2)*(-5/4)", "5/8"),
])
def test_subtasks_treat_all_rational_operands_as_atomic(expression, answer):
    from fractions import Fraction
    source = row(f"Calculate ({expression})+10.", str(Fraction(answer) + 10), "arithmetic__mixed")
    annotation = annotate_row(source, protocol="reasoning_v3")
    task = generate_subtask_rows(source, annotation, source_split="test", parent_row_index=0)[-1]
    trace = annotate_row(task, protocol="reasoning_v3")["scratchpad"]
    assert trace.count("call:") == 0
    assert task["module"] != "arithmetic__mixed"
    verify_annotation(task, task)
    for field, value in (("question", "Calculate 1 + 2."), ("call_result", "0"),
                         ("module", "arithmetic__mixed"), ("subtask_operation_count", 3)):
        with pytest.raises(ValueError, match="Atomic subtask"):
            annotate_row({**task, field: value}, protocol="reasoning_v2")


def test_reasoning_v3_algebra_delegates_final_division_and_verifier_rejects_edits():
    source = row("Solve -21*o + 32 + 120 = -37 for o.", "9", "algebra__linear_1d")
    annotation = annotate_row(source, protocol="reasoning_v3")
    trace = annotation["scratchpad"]
    assert "step:collect:" in trace
    assert "move:-21o=-189" in trace
    assert "call:divide(-189,-21) -> 9" in trace
    assert "verify:substitution satisfies the original equation" not in trace
    assert trace.endswith("o=9")
    verify_annotation(source, annotation)
    corrupted = dict(annotation, scratchpad=trace.replace("-> 9", "-> 8"))
    with pytest.raises(ValueError, match="reasoning_v3 calls"):
        verify_annotation(source, corrupted)


def test_subtask_generator_rejects_a_forged_whole_task_call():
    source = row("Calculate 25 + 32.", "57", "arithmetic__add_or_sub")
    forged = {"trace_version": "reasoning_v3", "scratchpad": "step:x\ncall:add(25,32) -> 57"}
    with pytest.raises(ValueError, match="not a proper subproblem"):
        generate_subtask_rows(source, forged, source_split="sft_train", parent_row_index=0)


def test_trace_verifier_rejects_corrupted_digit_and_substitution_steps() -> None:
    source = row("999 + 1", "1000", "arithmetic__add_or_sub")
    annotation = annotate_row(source)
    corrupted = dict(annotation, scratchpad=annotation["scratchpad"].replace("9,1,0:0,1", "9,1,0:1,1", 1))
    with pytest.raises(ValueError, match="transition"):
        verify_annotation(source, corrupted)
    decimal = row("0.2 + 9386818", "9386818.2", "arithmetic__add_or_sub")
    annotation = annotate_row(decimal)
    corrupted = dict(
        annotation,
        scratchpad=annotation["scratchpad"].replace("\nscale:1=>9386818.2", ""),
    )
    with pytest.raises(ValueError, match="scale annotation"):
        verify_annotation(decimal, corrupted)
    algebra = row("Solve -74*d + 5537428 = 5540240 for d.", "-38", "algebra__linear_1d")
    annotation = annotate_row(algebra)
    corrupted = dict(annotation, scratchpad=annotation["scratchpad"].replace("verify:5540240=5540240", "verify:5540241=5540240"))
    with pytest.raises(ValueError, match="substitution"):
        verify_annotation(algebra, corrupted)


def test_source_answer_mismatch_and_unsupported_syntax_fail_closed() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        annotate_row(row("2 + 2", "5", "arithmetic__add_or_sub"))
    with pytest.raises(ValueError):
        annotate_row(row("2 ** 3", "8", "arithmetic__mixed"))


def test_budget_counter_counts_full_prompt_completion_boundaries() -> None:
    item = row("47 + 28", "75", "arithmetic__add_or_sub")
    assert default_sequence_length(item, "+(47,28)\n7,8,0:5,1") == (
        1 + len("Q: 47 + 28\n") + 1 + len("\n+(47,28)\n7,8,0:5,1\n") + 1 + len("\nA: 75") + 1
    )
    assert default_sequence_length(item, "") >= len(item["question"]) + len(item["answer"])


def test_exact_budget_boundary_does_not_collapse_or_truncate() -> None:
    item = row("2 + 2", "4", "arithmetic__add_or_sub")
    expression, _ = _source_expression(item["question"], item["module"])
    builder = _Builder()
    _ExpressionParser(expression, builder).parse()
    scratch, collapsed, total = reduce_to_budget(
        item, builder.operations, max_tokens=512, length_fn=lambda _row, _scratch: 512
    )
    assert scratch and collapsed == {} and total == 512


def test_collapse_chooses_highest_priority_positive_saving_deterministically() -> None:
    item = row("12345 * 6789", "83810205", "arithmetic__mul")
    expression, _ = _source_expression(item["question"], item["module"])
    builder = _Builder()
    _ExpressionParser(expression, builder).parse()
    target = 500
    first = reduce_to_budget(item, deepcopy(builder.operations), max_tokens=target)
    second = reduce_to_budget(item, deepcopy(builder.operations), max_tokens=target)
    assert first == second
    assert first[1].get("addition", 0) >= 1
    assert first[2] <= target


def test_over_budget_skeleton_fails_without_truncation() -> None:
    item = row("2 + 2", "4", "arithmetic__add_or_sub")
    expression, _ = _source_expression(item["question"], item["module"])
    builder = _Builder()
    _ExpressionParser(expression, builder).parse()
    with pytest.raises(ValueError, match="skeleton exceeds"):
        reduce_to_budget(item, builder.operations, max_tokens=0)


def test_v2_verifier_replays_tokenizer_aware_collapse_budget() -> None:
    item = row(
        "What is 12345678901234567890 + 98765432109876543210?",
        "111111111011111111100",
        "arithmetic__add_or_sub",
    )
    length_fn = lambda example, scratchpad: default_sequence_length(example, scratchpad) + 300
    annotation = annotate_row(
        item, max_tokens=512, length_fn=length_fn, protocol="reasoning_v2"
    )

    assert annotation["reduced"] is True
    verify_annotation(item, annotation, length_fn=length_fn)


def test_v2_reducer_budgets_the_forward_trace_not_its_legacy_intermediate() -> None:
    item = row(
        "Calculate (-1/10) * (162649977/1000000000).",
        "-162649977/10000000000",
        "arithmetic__mixed",
    )

    def v3_character_length(example: dict[str, str], scratchpad: str) -> int:
        # reasoning_v3 encodes the two scratchpad delimiters as 11 characters;
        # v1/v2 count them as two reserved IDs.
        return default_sequence_length(example, scratchpad) + 9

    annotation = annotate_row(
        item, max_tokens=512, length_fn=v3_character_length, protocol="reasoning_v2"
    )

    assert annotation["retained_length"] == 507
    verify_annotation(item, annotation, length_fn=v3_character_length)
