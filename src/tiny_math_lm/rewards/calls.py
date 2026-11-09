"""Bounded, exact arithmetic shared by v3 validation and rollout execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

EXECUTOR_VERSION = 1
_SCALAR = r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?"
_REQUEST = re.compile(rf"call:(add|subtract|multiply|divide)\(({_SCALAR}),({_SCALAR})\) -> ")


class CallError(ValueError):
    """A request cannot safely be executed or validated."""


@dataclass(frozen=True)
class ArithmeticCall:
    operation: str
    left: str
    right: str
    result: str


def scalar_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(value)


def execute_call(request: str, *, max_scalar_chars: int = 128) -> ArithmeticCall:
    match = _REQUEST.fullmatch(request)
    if match is None:
        raise CallError("malformed_request")
    operation, left, right = match.groups()
    if max(len(left), len(right)) > max_scalar_chars:
        raise CallError("scalar_limit")
    a, b = Fraction(left), Fraction(right)
    if scalar_text(a) != left or scalar_text(b) != right:
        raise CallError("noncanonical_operand")
    if operation == "divide" and b == 0:
        raise CallError("division_by_zero")
    result = scalar_text({
        "add": lambda: a + b, "subtract": lambda: a - b,
        "multiply": lambda: a * b, "divide": lambda: a / b,
    }[operation]())
    if len(result) > max_scalar_chars:
        raise CallError("scalar_limit")
    return ArithmeticCall(operation, left, right, result)


def validate_call(line: str, *, max_scalar_chars: int = 128) -> ArithmeticCall:
    request, separator, result = line.partition(" -> ")
    if not separator:
        raise CallError("malformed_request")
    call = execute_call(request + separator, max_scalar_chars=max_scalar_chars)
    if result != call.result:
        raise CallError("incorrect_call_result")
    return call
