from tinymath.data.trace_explanations import explain_reasoning_trace


def test_algebra_trace_explains_collect_move_solution_and_check():
    trace = """t0=collect:0d+(1275)=-75d+(0)
move:75d=-1275
d=-17
verify:1275=1275"""

    explanation = explain_reasoning_trace(trace)

    assert any("Collect the variable and constant terms" in step for step in explanation)
    assert any("Move the variable terms" in step and "75d = -1275" in step for step in explanation)
    assert any("Divide both sides by 75" in step and "d = -17" in step for step in explanation)
    assert any("both sides equal 1275" in step for step in explanation)


def test_arithmetic_trace_explains_operation_and_carry():
    explanation = explain_reasoning_trace("t0=+(47,28)=75\n7,8,0:5,1")

    assert explanation[0] == "Add 47 and 28: 47 + 28 = 75."
    assert any("write 5 and carry 1" in step for step in explanation)


def test_subtraction_trace_explains_borrow_instead_of_carry():
    explanation = explain_reasoning_trace("t0=-(25,39)=-14\n5,9,0:6,1")

    assert any("Subtract digit 9 and borrow 0 from 5" in step for step in explanation)
    assert any("pass borrow 1" in step for step in explanation)


def test_division_trace_explains_long_division_and_reduction():
    explanation = explain_reasoning_trace("t0=/(-24,-66)=4/11\n2:0,2\n24:0,24\ngcd=6;4/11")

    assert any("Divide -24 by -66" in step for step in explanation)
    assert any("quotient digit 0" in step and "remainder 24" in step for step in explanation)
    assert any("greatest common divisor 6" in step for step in explanation)


def test_division_trace_explains_sign_cancellation():
    explanation = explain_reasoning_trace("t0=/(-12,-3)=4\n12:4,0\ngcd=3;4/1")

    assert "The two negative signs cancel, so the quotient is positive." in explanation


def test_multiplication_trace_explains_partial_product_and_carry():
    explanation = explain_reasoning_trace(
        "p0=-1*328*9*10^0=-2952\np1=M199809@2=sign:-262400\n8,0:4,6\n2,6:2,2\n3,2:6,2\ncarry:2"
    )

    assert any("multiply 328 by the multiplier digit 9" in step for step in explanation)
    assert any("multiply 328 by the 8 in the 3's place" in step for step in explanation)
    assert any("write 4, and carry 6" in step for step in explanation)
