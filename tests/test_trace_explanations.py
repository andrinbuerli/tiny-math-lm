from tinymath.data.trace_explanations import explain_reasoning_trace, transcribe_scratchpad


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
    assert any("multiply 328 by the 8 in the hundreds place" in step for step in explanation)
    assert any("write 4, and carry 6" in step for step in explanation)


def test_transcriber_supports_v2_algebra_and_forward_arithmetic():
    algebra = """step:collect:2x+(3)=0x+(9)
move:2x=6
divide both sides by 2
verify:substitution satisfies the original equation"""
    arithmetic = """step:+(1,2)
1,2,0:3,0
=>3
-(t0,4)
3,4,0:9,1"""

    algebra_steps = transcribe_scratchpad(algebra, version="reasoning_v2")
    arithmetic_steps = transcribe_scratchpad(arithmetic)

    assert any("Collect like terms" in step for step in algebra_steps)
    assert any("Divide both sides by 2" in step and "isolate x" in step for step in algebra_steps)
    assert any("check both sides match" in step for step in algebra_steps)
    assert arithmetic_steps[0] == "Add 1 and 2."
    assert any("This gives 3." in step for step in arithmetic_steps)
    assert any("Subtract 4 from step 1 (3)" in step for step in arithmetic_steps)
    assert any("pass borrow 1" in step for step in arithmetic_steps)


def test_v2_transcriber_handles_forward_multiplication_and_division():
    multiplication = """step:mul 34@0
2,0:8,0
1,0:4,0
=>48
mul 34@1
2,0:6,0
1,0:3,0
=>360
+(p0,p1)
8,0,0:8,0"""
    division = """step:/(15,4)
1:0,1
15:3,3
gcd=1"""

    multiplication_steps = transcribe_scratchpad(multiplication, version="v2")
    division_steps = transcribe_scratchpad(division, version="reasoning_v2")

    assert any("Multiply 12 by the 4 in the ones place" in step for step in multiplication_steps)
    assert any("partial product is 48" in step for step in multiplication_steps)
    assert any("step 1 (48)" in step and "step 2 (360)" in step for step in multiplication_steps)
    assert any("Divide 15 by 4" in step for step in division_steps)
    assert any("greatest common divisor, 1" in step for step in division_steps)


def test_transcriber_rejects_unknown_or_mismatched_trace_versions():
    try:
        transcribe_scratchpad("t0=+(1,2)=3", version="reasoning_v2")
    except ValueError as exc:
        assert "must start with a derivation step" in str(exc)
    else:
        raise AssertionError("A v1 trace should not be accepted as v2")

    try:
        transcribe_scratchpad("step:+(1,2)", version="v3")
    except ValueError as exc:
        assert "Unsupported reasoning trace version" in str(exc)
    else:
        raise AssertionError("Unknown trace versions should be rejected")
