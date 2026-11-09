"""Safely parse generated scalar math without evaluating arbitrary code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import sympy as sp


class AnswerFormatError(ValueError):
    """A generated completion is not a supported complete scalar answer."""


@dataclass(frozen=True)
class ParsedAnswer:
    expression: sp.Expr
    variable: str | None = None


def _from_ast(node: ast.AST, source: str, *, depth: int = 0) -> sp.Expr:
    if depth > 24:
        raise AnswerFormatError("answer expression is too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        literal = ast.get_source_segment(source, node)
        try:
            if literal and any(character in literal for character in ".eE"):
                return sp.Rational(str(Decimal(literal)))
            return sp.Integer(literal or node.value)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise AnswerFormatError("invalid numeric literal") from exc
    if isinstance(node, ast.Name) and node.id == "x":
        return sp.Symbol("x")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _from_ast(node.operand, source, depth=depth + 1)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _from_ast(node.left, source, depth=depth + 1)
        right = _from_ast(node.right, source, depth=depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise AnswerFormatError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow) and right.is_Integer and abs(int(right)) <= 16:
            return left**int(right)
    raise AnswerFormatError("unsupported syntax in scalar answer")


def parse_answer(text: str) -> ParsedAnswer:
    """Parse a full numeric/scalar answer or a simple `x = scalar` solution."""
    answer = text.strip()
    if not answer or len(answer) > 256 or "\n" in answer or "\r" in answer:
        raise AnswerFormatError("answer must be one nonempty line of at most 256 characters")
    if answer.count("=") > 1:
        raise AnswerFormatError("answer contains more than one equality")
    variable = None
    expression_text = answer
    if "=" in answer:
        left, expression_text = (part.strip() for part in answer.split("=", 1))
        if re.fullmatch(r"[A-Za-z]", left) is None:
            raise AnswerFormatError("only a simple one-letter variable assignment is supported")
        variable = left
    try:
        tree = ast.parse(expression_text, mode="eval")
    except SyntaxError as exc:
        raise AnswerFormatError("malformed scalar answer") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise AnswerFormatError("answer expression is too complex")
    expression = _from_ast(tree.body, expression_text)
    if expression.free_symbols - {sp.Symbol("x")}:
        raise AnswerFormatError("only the variable x is supported")
    return ParsedAnswer(expression=expression, variable=variable)


def answers_equal(prediction: str, target: str) -> bool:
    """Compare exact rationals and safely constructed scalar expressions."""
    try:
        predicted = parse_answer(prediction).expression
        expected = parse_answer(target).expression
        difference = sp.cancel(predicted - expected)
        return difference == 0 or difference.equals(0) is True
    except (AnswerFormatError, ArithmeticError, TypeError, ValueError, sp.SympifyError):
        return False


def verify_answer(
    prediction: str, target: str, *, question: str | None = None
) -> tuple[bool, bool]:
    """Return correctness and format validity for a complete generated answer."""
    try:
        parsed = parse_answer(prediction)
    except AnswerFormatError:
        return False, False
    if question is not None and parsed.variable is not None:
        expected = re.search(r"\bfor\s+([A-Za-z])\b", question)
        if expected is not None and parsed.variable != expected.group(1):
            return False, True
    return answers_equal(prediction, target), True
