"""Exact arithmetic references and deterministic scratchpad budget reduction.

The public :func:`annotate_row` accepts a source bundle row and returns a copy
with an inner-text ``scratchpad`` plus length/collapse provenance. Rendering is
kept separate from budgeting: callers may pass a tokenizer-aware ``length_fn``
which receives ``(row, scratchpad)`` and returns the complete sequence length.
The default counts one token per ordinary Unicode character and one token per
reserved boundary/BOS/EOS ID, using the Experiment 11 framing.

This deliberately implements a bounded allowlist of the five selected
Mathematics Dataset module grammars. It never evaluates Python expressions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction

MODULES = frozenset({
    "arithmetic__add_or_sub", "arithmetic__mul", "arithmetic__div",
    "arithmetic__mixed", "algebra__linear_1d",
})
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_NUMBER_RE = re.compile(NUMBER)
_VAR = re.compile(r"[a-z]")


def _fraction(text: str) -> Fraction:
    """Parse a signed decimal exactly as an integer ratio."""
    text = text.strip()
    rational = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", text)
    if rational:
        return Fraction(int(rational.group(1)), int(rational.group(2)))
    if "." not in text:
        return Fraction(int(text), 1)
    sign = -1 if text.startswith("-") else 1
    body = text.lstrip("+-")
    whole, _, fractional = body.partition(".")
    digits = (whole or "0") + fractional
    return Fraction(sign * int(digits or "0"), 10 ** len(fractional))


def _scalar(value: Fraction | int) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _decimal(value: Fraction) -> str:
    """Format a terminating decimal without floats; otherwise use a fraction."""
    den = value.denominator
    rest = den
    twos = fives = 0
    while rest % 2 == 0:
        rest //= 2; twos += 1
    while rest % 5 == 0:
        rest //= 5; fives += 1
    if rest != 1:
        return _scalar(value)
    scale = max(twos, fives)
    scaled = value.numerator * (2 ** (scale - twos)) * (5 ** (scale - fives))
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(scale + 1, "0")
    if scale:
        digits = digits[:-scale] + "." + digits[-scale:]
        digits = digits.rstrip("0").rstrip(".")
    return sign + digits


def _integer_digits(value: int) -> list[int]:
    return [int(ch) for ch in str(abs(value))]


@dataclass
class Operation:
    """A collapsible mathematical subtree with stable source order."""

    identifier: str
    category: str
    detailed: str
    collapsed: str
    order: int
    children: tuple[str, ...] = ()
    essential: bool = False
    collapsed_now: bool = False

    @property
    def current(self) -> str:
        return self.collapsed if self.collapsed_now else self.detailed


@dataclass
class _Trace:
    value: Fraction
    lines: list[str]
    operations: list[Operation]
    ref: str


def _add_lines(a: int, b: int, result: int) -> list[str]:
    """Right-to-left signed integer addition, including carry/borrow states."""
    # Rows in this compact form operate on magnitudes; sign handling is stated
    # by the exact result equation. Opposite signs are rendered by subtracting.
    if a < 0 and b < 0:
        lines = _add_lines(-a, -b, -result)
        return lines
    if (a < 0) != (b < 0):
        if a < 0:
            a, b = b, a
        return _sub_lines(a, -b, result)
    da, db = _integer_digits(a), _integer_digits(b)
    carry = 0
    rows: list[str] = []
    for ix in range(max(len(da), len(db))):
        x = da[-1 - ix] if ix < len(da) else 0
        y = db[-1 - ix] if ix < len(db) else 0
        total = x + y + carry
        out, carry_next = total % 10, total // 10
        rows.append(f"{x},{y},{carry}:{out},{carry_next}")
        carry = carry_next
    if carry:
        rows.append(f"carry:{carry}")
    return rows


def _sub_lines(a: int, b: int, result: int) -> list[str]:
    """Magnitude subtraction with explicit cascading borrow transitions."""
    if a < b:
        return _sub_lines(b, a, -result)
    da, db = _integer_digits(a), _integer_digits(b)
    borrow = 0
    rows: list[str] = []
    for ix in range(len(da)):
        x = da[-1 - ix]
        y = db[-1 - ix] if ix < len(db) else 0
        raw = x - y - borrow
        borrow_next = int(raw < 0)
        out = raw + (10 if borrow_next else 0)
        rows.append(f"{x},{y},{borrow}:{out},{borrow_next}")
        borrow = borrow_next
    return rows


def _scaled_integer_pair(a: Fraction, b: Fraction) -> tuple[int, int, int]:
    scale = max(_scale(a), _scale(b))
    factor = 10**scale
    return int(a * factor), int(b * factor), scale


def _scale(value: Fraction) -> int:
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2; twos += 1
    while den % 5 == 0:
        den //= 5; fives += 1
    return max(twos, fives) if den == 1 else 0


class _Builder:
    def __init__(self) -> None:
        self.operations: list[Operation] = []
        self.lines: list[str] = []
        self.counter = 0
        self.partial_counter = 0

    def op(self, category: str, detailed: str, collapsed: str,
           *, children: Iterable[str] = (), essential: bool = False,
           identifier: str | None = None) -> str:
        ref = identifier or f"t{self.counter}"
        if identifier is None:
            self.counter += 1
        self.operations.append(Operation(ref, category, detailed, collapsed,
                                         len(self.operations), tuple(children), essential))
        self.lines.append(f"{ref}={detailed}")
        return ref

    def arithmetic(
        self, symbol: str, a: Fraction, b: Fraction,
        refs: tuple[str, str] | None = None,
    ) -> _Trace:
        if symbol == "+":
            result = a + b
        elif symbol == "-":
            result = a - b
        elif symbol == "*":
            result = a * b
        elif symbol == "/":
            if b == 0:
                raise ValueError("Division by zero in source question")
            result = a / b
        else:
            raise ValueError(f"Unsupported operation {symbol!r}")
        name = {"+": "addition", "-": "addition", "*": "multiplication", "/": "division"}[symbol]
        detail: list[str] = []
        if symbol in "+-":
            sa, sb = _scale(a), _scale(b)
            scale = max(sa, sb)
            expression = f"{symbol}({_decimal(a)},{_decimal(b)})={_decimal(result)}"
            # Fractions with non-terminating decimal expansions have no finite
            # base-10 digit trace. Keep their exact rational operation as the
            # mathematical step instead of truncating operands with int().
            has_nonterminating_operand = any(
                value.denominator != 1 and _scale(value) == 0 for value in (a, b)
            )
            if has_nonterminating_operand:
                detail_text = expression
            else:
                factor = 10**scale
                ai, bi, ri = int(a * factor), int(b * factor), int(result * factor)
                detail = _add_lines(ai, bi, ri) if symbol == "+" else _add_lines(ai, -bi, ri)
                detail_text = expression + "\n" + "\n".join(detail)
                if scale:
                    detail_text += f"\nscale:{scale}=>{_decimal(result)}"
        elif symbol == "*" and (
            a.denominator == 1 or _scale(a) > 0
        ) and (b.denominator == 1 or _scale(b) > 0):
            sa, sb = _scale(a), _scale(b)
            ai, bi = int(a * (10**sa)), int(b * (10**sb))
            lhs, rhs = abs(ai), abs(bi)
            parts: list[int] = []
            partial_ids: list[str] = []
            sign = -1 if result < 0 else 1
            for shift, digit in enumerate(reversed(_integer_digits(rhs))):
                carry = 0
                partial = 0
                emitted = []
                for d in reversed(_integer_digits(lhs)):
                    carry_in = carry
                    product = d * digit + carry
                    out, carry = product % 10, product // 10
                    partial += out * (10**len(emitted))
                    emitted.append(f"{d},{carry_in}:{out},{carry}")
                if carry:
                    emitted.append(f"carry:{carry}")
                    partial += carry * 10 ** len(_integer_digits(lhs))
                partial *= 10**shift
                signed_partial = partial * sign
                parts.append(signed_partial)
                partial_id = f"p{self.partial_counter}"
                self.partial_counter += 1
                partial_ids.append(partial_id)
                partial_lines = []
                if shift == 0:
                    # Record the multiplier digits in the right-to-left shift
                    # order used by the following partial-product rows.
                    partial_lines.append(
                        "digit:" + ",".join(str(digit) for digit in reversed(_integer_digits(rhs)))
                    )
                partial_lines.append(f"M{_decimal(b)}@{shift}={signed_partial}")
                if sign < 0:
                    partial_lines[0] = f"M{_decimal(b)}@{shift}=sign:{signed_partial}"
                partial_lines.extend(emitted)
                self.op(
                    "multiplication",
                    "\n".join(partial_lines),
                    f"{sign}*{lhs}*{digit}*10^{shift}={signed_partial}",
                    identifier=partial_id,
                )
            running = parts[0] if parts else 0
            prior_ref = partial_ids[0]
            for index, part in enumerate(parts[1:], start=1):
                next_value = running + part
                partial_ref = partial_ids[index]
                detail_header = (
                    f"{prior_ref}+{partial_ref} ->"
                    if index == len(parts) - 1
                    else f"{prior_ref}+{partial_ref}={next_value}"
                )
                addition_detail = (detail_header + "\n" +
                                   "\n".join(_add_lines(running, part, next_value)))
                prior_ref = self.op("addition", addition_detail,
                                    f"{prior_ref}+{partial_ref}={next_value}",
                                    children=(prior_ref,))
                running = next_value
            scale = sa + sb
            if scale:
                scaled_value = Fraction(running, 10**scale)
                scale_detail = f"scale:{running}/{10**scale}={_decimal(scaled_value)}"
                prior_ref = self.op(
                    "multiplication", scale_detail,
                    f"{prior_ref}/{10**scale}={_decimal(scaled_value)}",
                )
            return _Trace(result, [], self.operations, prior_ref)
        elif symbol == "*":
            detail_text = f"*({_scalar(a)},{_scalar(b)})={_scalar(result)}"
        elif symbol == "/":
            # Put both original operands over their least common denominator,
            # preserving their ratio while retaining the source dividend and
            # divisor for the long-division trace (for example, 12/3, not 4/1).
            ratio = a / b
            common_scale = math.lcm(a.denominator, b.denominator)
            source_dividend = a.numerator * (common_scale // a.denominator)
            source_divisor = b.numerator * (common_scale // b.denominator)
            numerator, denominator = abs(source_dividend), abs(source_divisor)
            rem = 0
            rows = [f"/({_scalar(a)},{_scalar(b)})={_scalar(ratio)}"]
            for digit in str(numerator):
                brought = rem * 10 + int(digit)
                qdigit, rem = divmod(brought, denominator)
                rows.append(f"{brought}:{qdigit},{rem}")
            gcd = _gcd(numerator, denominator)
            reduced_numerator = numerator // gcd
            reduced_denominator = denominator // gcd
            if source_dividend * source_divisor < 0:
                reduced_numerator = -reduced_numerator
            rows.append(f"gcd={gcd};{reduced_numerator}/{reduced_denominator}")
            detail_text = "\n".join(rows)
            if ratio < 0:
                detail_text = "sign:-\n" + detail_text
        else:
            detail_text = f"{symbol}({_scalar(a)},{_scalar(b)})={_scalar(result)}"
        ref_a, ref_b = refs or (_scalar(a), _scalar(b))
        has_reference = bool(
            refs and (ref_a.lstrip("+-").startswith("t") or ref_b.lstrip("+-").startswith("t"))
        )
        collapsed = (
            f"{ref_a}{symbol}{ref_b}={_scalar(result)}"
            if has_reference
            else f"{symbol}({_scalar(a)},{_scalar(b)})={_scalar(result)}"
        )
        if has_reference:
            lines = detail_text.splitlines()
            for index, line in enumerate(lines):
                if line.startswith(f"{symbol}("):
                    lines[index] = f"{symbol}({ref_a},{ref_b})={_scalar(result)}"
                    detail_text = "\n".join(lines)
                    break
        category = "single_digit" if max(abs(a.numerator), abs(b.numerator)) < 10 else name
        ref = self.op(category, detail_text, collapsed)
        return _Trace(result, [], self.operations, ref)


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _lex_expression(text: str) -> list[str]:
    tokens = re.findall(r"\d+(?:\.\d*)?|\.\d+|[()+*/-]", text)
    if "".join(tokens).replace(".", "", 1) == "":
        raise ValueError("No arithmetic expression")
    # Ensure every non-space input character belongs to a token.
    compact = re.sub(r"\s+", "", text)
    if "".join(tokens) != compact:
        raise ValueError(f"Expression contains unsupported syntax: {text!r}")
    return tokens


class _ExpressionParser:
    def __init__(self, text: str, builder: _Builder) -> None:
        self.tokens = _lex_expression(text)
        self.i = 0
        self.builder = builder

    def parse(self) -> _Trace:
        value = self._sum()
        if self.i != len(self.tokens):
            raise ValueError("Unexpected expression suffix")
        return value

    def _sum(self) -> _Trace:
        left = self._product()
        while self.i < len(self.tokens) and self.tokens[self.i] in ("+", "-"):
            op = self.tokens[self.i]; self.i += 1
            right = self._product()
            left = self.builder.arithmetic(op, left.value, right.value, (left.ref, right.ref))
        return left

    def _product(self) -> _Trace:
        left = self._atom()
        while self.i < len(self.tokens) and self.tokens[self.i] in ("*", "/"):
            op = self.tokens[self.i]; self.i += 1
            right = self._atom()
            left = self.builder.arithmetic(op, left.value, right.value, (left.ref, right.ref))
        return left

    def _atom(self) -> _Trace:
        if self.i >= len(self.tokens):
            raise ValueError("Missing expression atom")
        token = self.tokens[self.i]
        if token in ("+", "-"):
            self.i += 1
            atom = self._atom()
            value = atom.value if token == "+" else -atom.value
            ref = atom.ref if token == "+" else (
                f"-{atom.ref}" if atom.ref.startswith("t") else _scalar(value)
            )
            return _Trace(value, [], atom.operations, ref)
        if token == "(":
            self.i += 1
            atom = self._sum()
            if self.i >= len(self.tokens) or self.tokens[self.i] != ")":
                raise ValueError("Unclosed expression parenthesis")
            self.i += 1
            return atom
        self.i += 1
        value = _fraction(token)
        return _Trace(value, [], self.builder.operations, _scalar(value))


def _source_expression(question: str, module: str) -> tuple[str, str | None]:
    """Extract allowlisted expression text, and an optional operation hint."""
    text = question.strip().rstrip("?.")
    if module == "algebra__linear_1d":
        match = re.fullmatch(r"Solve\s+(.+?)\s*=\s*(.+?)\s+for\s+([a-z])", text, re.IGNORECASE)
        if not match:
            raise ValueError(f"Unsupported linear equation template: {question!r}")
        return f"{match.group(1)}={match.group(2)}", match.group(3)
    if module == "arithmetic__mixed":
        text = re.sub(r"^(?:Calculate|Evaluate|What is the value of|What is)\s+", "", text, flags=re.IGNORECASE)
        return text, None
    if module == "arithmetic__add_or_sub":
        sub = re.fullmatch(r"Subtract\s+(" + NUMBER + r")\s+from\s+(" + NUMBER + r")", text, re.IGNORECASE)
        if sub:
            return f"{sub.group(2)}-{sub.group(1)}", None
        text = re.sub(r"^(?:Work out|What is|Calculate|Add|Total of|Put together)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^together\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^Sum\s+", "", text, flags=re.IGNORECASE)
        distance = re.fullmatch(r"(?:the\s+)?(?:difference|distance) between\s*(" + NUMBER + r")\s+and\s+(" + NUMBER + r")", text, re.IGNORECASE)
        if distance:
            left, right = _fraction(distance.group(1)), _fraction(distance.group(2))
            left_abs, right_abs = abs(left), abs(right)
            if (left < 0) != (right < 0):
                return f"{_decimal(left_abs)}+{_decimal(right_abs)}", None
            high, low = sorted((left_abs, right_abs), reverse=True)
            return f"{_decimal(high)}-{_decimal(low)}", None
        less = re.fullmatch(r"(" + NUMBER + r")\s+less than\s+(" + NUMBER + r")", text, re.IGNORECASE)
        if less:
            return f"{less.group(2)}-{less.group(1)}", None
        text = re.sub(r"\s+take away\s+", "-", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+less than\s+", "-", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+plus\s+", "+", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+and\s+", "+", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+minus\s+", "-", text, flags=re.IGNORECASE)
        return text, None
    if module == "arithmetic__mul":
        text = re.sub(r"^(?:What is the product of|Product of|Multiply|Calculate|Work out|What is)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(?:and|times)\s+", "*", text, flags=re.IGNORECASE)
        return text, None
    if module == "arithmetic__div":
        text = re.sub(r"^(?:Calculate|What is|Divide)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+divided by\s+", "/", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+by\s+", "/", text, flags=re.IGNORECASE)
        return text, None
    raise ValueError(f"Unsupported module: {module!r}")


@dataclass(frozen=True)
class Linear:
    coefficient: Fraction
    constant: Fraction

    def __add__(self, other: Linear) -> Linear:
        return Linear(self.coefficient + other.coefficient, self.constant + other.constant)

    def __sub__(self, other: Linear) -> Linear:
        return Linear(self.coefficient - other.coefficient, self.constant - other.constant)


class _LinearParser(_ExpressionParser):
    def __init__(self, text: str, variable: str, builder: _Builder) -> None:
        tokens = re.findall(r"\d+(?:\.\d*)?|\.\d+|[a-z]|[()+*/-]", text)
        if "".join(tokens) != re.sub(r"\s+", "", text):
            raise ValueError("Unsupported linear expression syntax")
        self.tokens, self.i, self.builder, self.variable = tokens, 0, builder, variable

    def parse_linear(self) -> Linear:
        value = self._linear_sum()
        if self.i != len(self.tokens):
            raise ValueError("Unexpected linear expression suffix")
        return value

    def _linear_sum(self) -> Linear:
        left = self._linear_product()
        while self.i < len(self.tokens) and self.tokens[self.i] in ("+", "-"):
            op = self.tokens[self.i]; self.i += 1
            right = self._linear_product()
            left = left + right if op == "+" else left - right
        return left

    def _linear_product(self) -> Linear:
        left = self._linear_atom()
        while self.i < len(self.tokens) and self.tokens[self.i] in ("*", "/"):
            op = self.tokens[self.i]; self.i += 1
            right = self._linear_atom()
            if op == "*":
                if left.coefficient and right.coefficient:
                    raise ValueError("Nonlinear equation")
                left = Linear(left.coefficient * right.constant + right.coefficient * left.constant,
                              left.constant * right.constant)
            else:
                if right.coefficient or right.constant == 0:
                    raise ValueError("Variable or zero divisor in linear equation")
                left = Linear(left.coefficient / right.constant, left.constant / right.constant)
        return left

    def _linear_atom(self) -> Linear:
        if self.i >= len(self.tokens):
            raise ValueError("Missing equation atom")
        token = self.tokens[self.i]
        if token in ("+", "-"):
            self.i += 1
            value = self._linear_atom()
            return value if token == "+" else Linear(-value.coefficient, -value.constant)
        if token == "(":
            self.i += 1
            value = self._linear_sum()
            if self.i >= len(self.tokens) or self.tokens[self.i] != ")":
                raise ValueError("Unclosed equation parenthesis")
            self.i += 1
            return value
        self.i += 1
        if token == self.variable:
            return Linear(Fraction(1), Fraction(0))
        if token.isalpha():
            raise ValueError("Unexpected variable in linear equation")
        return Linear(Fraction(0), _fraction(token))


def _linear_trace(question: str, answer: str) -> tuple[str, list[Operation], Fraction]:
    equation, variable = _source_expression(question, "algebra__linear_1d")
    lhs_text, rhs_text = equation.split("=", 1)
    builder = _Builder()
    lhs = _LinearParser(lhs_text, variable, builder).parse_linear()
    rhs = _LinearParser(rhs_text, variable, builder).parse_linear()
    coeff, const = lhs.coefficient - rhs.coefficient, rhs.constant - lhs.constant
    if coeff == 0:
        raise ValueError("Equation does not determine a unique solution")
    result = const / coeff
    expected = _fraction(answer)
    if result != expected:
        raise ValueError(f"Derived answer {_scalar(result)} disagrees with label {answer!r}")
    lines = [(f"collect:{_scalar(lhs.coefficient)}{variable}+({_scalar(lhs.constant)})="
              f"{_scalar(rhs.coefficient)}{variable}+({_scalar(rhs.constant)})"),
             f"move:{_scalar(coeff)}{variable}={_scalar(const)}",
             f"{variable}={_scalar(result)}",
             (f"verify:{_scalar(lhs.coefficient * result + lhs.constant)}="
              f"{_scalar(rhs.coefficient * result + rhs.constant)}")]
    # Algebra transformations are essential; arithmetic helper traces can be
    # collapsed before them, while this final verification stays visible.
    builder.op("algebra", "\n".join(lines), f"solve({equation})={variable}:{_scalar(result)}",
               essential=True)
    return "\n".join(f"t{index}={op.detailed}" for index, op in enumerate(builder.operations)), builder.operations, result


def render_operations(operations: list[Operation]) -> str:
    """Render active operation nodes in original deterministic order."""
    return "\n".join(f"{op.identifier}={op.current}" for op in operations)


def default_sequence_length(row: dict[str, str], scratchpad: str) -> int:
    """Count full Experiment 11 framing with one token per ordinary character."""
    # BOS + ``Q: {question}\n``; the completion contains two structural IDs,
    # a newline on each side of the inner scratchpad, ``\nA: ``, answer, EOS.
    return (
        1
        + len("Q: " + row["question"] + "\n")
        + 1
        + len("\n" + scratchpad + "\n")
        + 1
        + len("\nA: " + row["answer"])
        + 1
    )


def _collapse_order(op: Operation) -> tuple[int, int]:
    rank = {"single_digit": 0, "addition": 1, "multiplication": 2,
            "division": 3, "mixed": 4, "algebra": 5}
    return rank.get(op.category, 6), op.order


def reduce_to_budget(
    row: dict[str, str], operations: list[Operation], *, max_tokens: int = 512,
    length_fn: Callable[[dict[str, str], str], int] = default_sequence_length,
) -> tuple[str, dict[str, int], int]:
    """Collapse whole operation subtrees by category and positive token savings.

    At each iteration the highest priority available category wins; among its
    nodes, the largest positive measured saving wins, then original order.
    Essential algebra nodes and the last visible mathematical step are kept.
    """
    if not operations:
        raise ValueError("At least one mathematical operation is required")
    counts: dict[str, int] = {}
    scratch = render_operations(operations)
    total = length_fn(row, scratch)
    while total > max_tokens:
        candidates = []
        visible = [op for op in operations if not op.collapsed_now]
        priorities = sorted({_collapse_order(op)[0] for op in visible if not op.essential})
        selected = None
        for rank in priorities:
            group = [op for op in visible if _collapse_order(op)[0] == rank and not op.essential]
            for op in group:
                old = op.collapsed_now
                op.collapsed_now = True
                candidate_text = render_operations(operations)
                saving = total - length_fn(row, candidate_text)
                op.collapsed_now = old
                if saving > 0:
                    candidates.append((saving, -op.order, op, candidate_text))
            if candidates:
                selected = max(candidates, key=lambda item: (item[0], item[1]))
                break
        if selected is None:
            raise ValueError(f"Fully collapsed scratchpad skeleton exceeds {max_tokens} tokens: {total}")
        _, _, op, _ = selected
        op.collapsed_now = True
        counts[op.category] = counts.get(op.category, 0) + 1
        scratch = render_operations(operations)
        total = length_fn(row, scratch)
    if total > max_tokens:
        raise ValueError(f"Fully collapsed scratchpad skeleton exceeds {max_tokens} tokens: {total}")
    if not scratch.strip():
        raise ValueError("Budget reduction removed every mathematical step")
    return scratch, counts, total


def annotate_row(
    row: dict[str, str], *, max_tokens: int = 512,
    length_fn: Callable[[dict[str, str], str], int] = default_sequence_length,
    protocol: str = "reasoning_v1",
) -> dict[str, str | int | dict[str, int]]:
    """Verify and annotate a source row without mutating it."""
    if protocol not in {"reasoning_v1", "reasoning_v2"}:
        raise ValueError(f"Unsupported trace protocol: {protocol}")
    if not MODULES.__contains__(row.get("module", "")):
        raise ValueError(f"Unsupported module: {row.get('module')!r}")
    answer = row["answer"].strip()
    expected = _fraction(answer)
    module = row["module"]
    if module == "algebra__linear_1d":
        _, operations, value = _linear_trace(row["question"], answer)
        if value != expected:
            raise ValueError("Linear result verification failed")
    else:
        expression, _ = _source_expression(row["question"], module)
        # Word templates normalize to operators; signed numbers remain exact.
        expression = re.sub(r"(?<=\d)\s+(?=[+-]?\d)", " ", expression)
        builder = _Builder()
        trace = _ExpressionParser(expression, builder).parse()
        value = trace.value
        if value != expected:
            raise ValueError(f"Derived answer {_scalar(value)} disagrees with label {answer!r}")
        operations = builder.operations
        invalid_count = (len(operations) < 1 if module == "arithmetic__mul" else len(operations) != 1)
        if module in ("arithmetic__add_or_sub", "arithmetic__mul", "arithmetic__div") and invalid_count:
            # Simple module grammars should represent one binary operation.
            raise ValueError(f"Unexpected operation count for {module}: {len(operations)}")
    if not operations:
        raise ValueError("No structured mathematical operation was produced")
    # Module-aware collapse categories preserve priority requested by the spec.
    if module == "arithmetic__mixed":
        for op in operations:
            if op.category not in ("single_digit", "algebra"):
                op.category = "mixed"
    full_text = render_operations(operations)
    if protocol == "reasoning_v2":
        full_text = _forward_trace(full_text, operations, value)
    full_length = length_fn(row, full_text)
    scratch, collapsed, retained_length = reduce_to_budget(
        row, operations, max_tokens=max_tokens, length_fn=length_fn)
    if protocol == "reasoning_v2":
        scratch = _forward_trace(scratch, operations, value)
        retained_length = length_fn(row, scratch)
        while retained_length > max_tokens:
            candidates: list[tuple[int, int, Operation, str]] = []
            for op in operations:
                if op.collapsed_now or op.essential:
                    continue
                op.collapsed_now = True
                candidate = _forward_trace(render_operations(operations), operations, value)
                saving = retained_length - length_fn(row, candidate)
                op.collapsed_now = False
                if saving > 0:
                    candidates.append((saving, -op.order, op, candidate))
            if not candidates:
                raise ValueError(
                    f"Forward trace skeleton exceeds {max_tokens} tokens: {retained_length}"
                )
            _, _, chosen, scratch = max(candidates, key=lambda item: (item[0], item[1]))
            chosen.collapsed_now = True
            collapsed[chosen.category] = collapsed.get(chosen.category, 0) + 1
            retained_length = length_fn(row, scratch)
    if _fraction(answer) != value:
        raise AssertionError("Final scalar changed during annotation")
    annotated = dict(row)
    annotated.update({
        "scratchpad": scratch,
        "full_length": full_length,
        "retained_length": retained_length,
        "reduced": full_length > retained_length,
        "collapsed_categories": collapsed,
        "collapsed_count": sum(collapsed.values()),
    })
    if protocol == "reasoning_v2":
        annotated["trace_version"] = protocol
    return annotated


_DEFINITION = re.compile(r"^([tp]\d+)=(.*)$")
_TRANSITION = re.compile(r"^(\d+),(\d+),(\d+):(\d+),(\d+)$")


def _verify_addition_rows(a: int, b: int, rows: list[str]) -> int:
    """Independently replay the emitted right-to-left carry/borrow rows."""
    rows = [row for row in rows if not row.startswith("scale:")]
    if a < 0 and b < 0:
        return -_verify_addition_rows(-a, -b, rows)
    if (a < 0) != (b < 0):
        minuend, subtrahend = (b, -a) if a < 0 else (a, -b)
        return _verify_subtraction_rows(minuend, subtrahend, rows)
    da, db = _integer_digits(a), _integer_digits(b)
    carry = 0
    out_digits = []
    row_index = 0
    for index in range(max(len(da), len(db))):
        x = da[-1 - index] if index < len(da) else 0
        y = db[-1 - index] if index < len(db) else 0
        total = x + y + carry
        digit, next_carry = total % 10, total // 10
        if row_index >= len(rows) or rows[row_index] != f"{x},{y},{carry}:{digit},{next_carry}":
            raise ValueError("Addition carry transition does not replay")
        out_digits.append(digit)
        carry = next_carry
        row_index += 1
    if carry:
        if row_index >= len(rows) or rows[row_index] != f"carry:{carry}":
            raise ValueError("Addition final carry is missing or incorrect")
        out_digits.append(carry)
        row_index += 1
    if row_index != len(rows):
        raise ValueError("Addition trace has unexpected transition rows")
    return sum(digit * 10**index for index, digit in enumerate(out_digits))


def _forward_trace(legacy: str, operations: list[Operation], final_value: Fraction) -> str:
    """Convert v1 result-first operation blocks to v2 forward derivations.

    A step's operands and digit work come first. Nonterminal intermediate values
    are emitted after their derivation because later steps may use them. The
    terminal scalar is never emitted in the scratchpad.
    """
    final_name = operations[-1].identifier if operations else ""
    lines: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    for line in legacy.splitlines():
        match = _DEFINITION.fullmatch(line)
        if match:
            blocks.append((match.group(1), [match.group(2)]))
        elif blocks:
            blocks[-1][1].append(line)
        else:
            raise ValueError("Malformed source trace while producing reasoning_v2")
    operation_by_name = {op.identifier: op for op in operations}
    for name, block in blocks:
        head = block[0]
        offset = 0
        if head == "sign:-":
            lines.append("step:sign negative")
            offset = 1
            head = block[offset]
        if head.startswith("collect:"):
            lines.extend("step:" + item for item in block if item.startswith(("collect:", "move:")))
            continue

        multiplication_header = next(
            (item for item in block[offset:] if item.startswith("M") and "@" in item), None
        )
        partial = re.fullmatch(r"(-?1)\*(\d+)\*(\d+)\*10\^(\d+)=(-?\d+)", head)
        if partial:
            sign, multiplicand, digit, shift, answer = partial.groups()
            lines.append(f"step:mul {multiplicand}*{digit}@{shift}")
            result = Fraction(int(answer))
            if int(sign) < 0 and result > 0:
                result = -result
        elif multiplication_header is not None:
            match = re.fullmatch(r"M(.+?)@(\d+)=(?:sign:)?(-?\d+)", multiplication_header)
            if not match:
                raise ValueError("Malformed multiplication step")
            multiplier, shift, answer = match.groups()
            lines.append(f"step:mul {multiplier}@{shift}")
            for item in block[offset:]:
                if item.startswith("digit:") or item == multiplication_header:
                    continue
                if item.startswith("carry:") or _TRANSITION.fullmatch(item) or re.fullmatch(r"\d+,\d+:\d+,\d+", item):
                    lines.append(item)
            result = Fraction(int(answer))
        elif head.startswith(("+(", "-(", "*(", "/(")):
            match = re.fullmatch(r"([+*/-])\(([^,]+),([^)]*)\)=([^\s]+)", head)
            if not match:
                raise ValueError("Malformed arithmetic step")
            symbol, left, right, answer = match.groups()
            lines.append(f"step:{symbol}({left},{right})")
            result = _fraction(answer)
            for item in block[offset + 1:]:
                if item.startswith("scale:"):
                    scale = re.match(r"scale:(\d+)", item)
                    if scale:
                        lines.append("scale:" + scale.group(1))
                elif item.startswith("gcd="):
                    lines.append(item.split(";", 1)[0])
                else:
                    lines.append(item)
        else:
            scale_head = re.fullmatch(r"scale:(-?\d+)/(\d+)=(-?\d+(?:\.\d+)?)", head)
            if scale_head is None:
                scale_head = re.fullmatch(r"([tp]\d+)/(\d+)=(-?\d+(?:\.\d+)?)", head)
            if scale_head:
                source, denominator, answer = scale_head.groups()
                if source.startswith(("t", "p")):
                    prior_name = source
                else:
                    operation_index = [op.identifier for op in operations].index(name)
                    prior_name = operations[operation_index - 1].identifier if operation_index else ""
                lines.append(f"step:scale {prior_name} by {denominator}")
                result = _fraction(answer)
            else:
                match = re.fullmatch(r"([^=]+)=(-?\d+(?:/\d+)?)", head)
                if match:
                    expression, answer = match.groups()
                    lines.append("step:" + expression)
                    result = _fraction(answer)
                elif re.fullmatch(r"[+-]?[tp]\d+\+[+-]?[tp]\d+ ->", head):
                    left, right = head[:-3].split("+")
                    lines.append(f"step:+({left},{right})")
                    result = _fraction(operation_by_name[name].collapsed.rsplit("=", 1)[-1])
                    lines.extend(block[1:])
                else:
                    match = re.fullmatch(r"([+-]?[tp]\d+)([+*/-])([+-]?[tp]\d+)=(-?\d+(?:/\d+)?)", head)
                    if not match:
                        raise ValueError(f"Cannot convert operation {name}: {head!r}")
                    left, symbol, right, answer = match.groups()
                    lines.append(f"step:{symbol}({left},{right})")
                    result = _fraction(answer)
        if name != final_name and result != final_value:
            lines.append(f"=>{_scalar(result)}")
    step_seen = False
    compact_lines = []
    for line in lines:
        if line.startswith("step:"):
            compact_lines.append(line if not step_seen else line[5:])
            step_seen = True
        else:
            compact_lines.append(line)
    rendered = "\n".join(compact_lines).strip()
    if not rendered:
        raise ValueError("reasoning_v2 cannot have an empty trace")
    if any(
        (match := re.fullmatch(r"=>(-?\d+(?:/\d+)?)", line))
        and _fraction(match.group(1)) == final_value
        for line in rendered.splitlines()
    ):
        raise ValueError("Final answer leaked into a scratchpad result line")
    return rendered


def _verify_forward_annotation(row: dict[str, str], scratchpad: str) -> None:
    """Replay a v2 trace by rebuilding the exact source derivation and budget path."""
    # The established verifier independently derives the source value. The v2
    # comparison additionally rejects omitted, reordered, result-first, or
    # edited steps by comparing against a freshly regenerated forward trace.
    expected_v1 = annotate_row(row, max_tokens=512, protocol="reasoning_v1")
    verify_annotation(row, expected_v1)
    expected_v2 = annotate_row(row, max_tokens=512, protocol="reasoning_v2")
    if scratchpad != expected_v2["scratchpad"]:
        raise ValueError("reasoning_v2 trace is malformed, result-first, incomplete, or inconsistent")


def _verify_decimal_scale(a: Fraction, b: Fraction, result: Fraction, rows: list[str]) -> list[str]:
    scale = max(_scale(a), _scale(b))
    scale_lines = [line for line in rows if line.startswith("scale:")]
    if scale:
        expected_line = f"scale:{scale}=>{_decimal(result)}"
        if scale_lines != [expected_line]:
            raise ValueError(f"Decimal scale annotation must be exactly {expected_line!r}")
    elif scale_lines:
        raise ValueError("Unexpected decimal scale annotation for integer arithmetic")
    return [line for line in rows if not line.startswith("scale:")]


def _verify_subtraction_rows(a: int, b: int, rows: list[str]) -> int:
    rows = [row for row in rows if not row.startswith("scale:")]
    if a < b:
        return -_verify_subtraction_rows(b, a, rows)
    da, db = _integer_digits(a), _integer_digits(b)
    borrow = 0
    out_digits = []
    if len(rows) != len(da):
        raise ValueError("Subtraction trace has the wrong number of digit rows")
    for index, row in enumerate(rows):
        x = da[-1 - index]
        y = db[-1 - index] if index < len(db) else 0
        raw = x - y - borrow
        next_borrow = int(raw < 0)
        digit = raw + (10 if next_borrow else 0)
        if row != f"{x},{y},{borrow}:{digit},{next_borrow}":
            raise ValueError("Subtraction borrow transition does not replay")
        out_digits.append(digit)
        borrow = next_borrow
    return sum(digit * 10**index for index, digit in enumerate(out_digits))


def _verify_multiplier_rows(
    multiplier: Fraction, shift: int, lines: list[str], multiplicand: Fraction | None = None
) -> int:
    scale = _scale(multiplier)
    multiplier_integer = abs(int(multiplier * 10**scale))
    digits = _integer_digits(multiplier_integer)
    if shift >= len(digits):
        multiplier_digit = 0
    else:
        multiplier_digit = digits[-1 - shift]
    if multiplicand is not None:
        multiplicand_scale = _scale(multiplicand)
        multiplicand_integer = abs(int(multiplicand * 10**multiplicand_scale))
        expected_digits = list(reversed(_integer_digits(multiplicand_integer)))
        actual_digits = [
            int(match.group(1))
            for line in lines
            if (match := re.fullmatch(r"(\d+),(\d+):(\d+),(\d+)", line))
        ]
        if actual_digits != expected_digits:
            raise ValueError("Multiplication rows do not use the source multiplicand digits")
    previous_carry = 0
    stated_final_carry: int | None = None
    for line in lines:
        if line.startswith(("M", "digit:")):
            continue
        if line.startswith("carry:"):
            stated_final_carry = int(line.split(":", 1)[1])
            continue
        match = re.fullmatch(r"(\d+),(\d+):(\d+),(\d+)", line)
        if not match:
            raise ValueError(f"Malformed multiplication transition: {line!r}")
        digit, carry_in, result_digit, carry_out = map(int, match.groups())
        product = digit * multiplier_digit + carry_in
        if product % 10 != result_digit or product // 10 != carry_out:
            raise ValueError("Multiplication digit/carry transition does not replay")
        if carry_in != previous_carry:
            raise ValueError("Multiplication carry chain is discontinuous")
        previous_carry = carry_out
    if (stated_final_carry or 0) != previous_carry:
        raise ValueError("Multiplication final carry is missing or incorrect")
    # Re-read rows to collect result digits in their emitted least-significant
    # first order, then append the explicit final carry when present.
    result_digits = []
    final_carry = 0
    for line in lines:
        if line.startswith("carry:"):
            final_carry = int(line.split(":", 1)[1])
            continue
        match = re.fullmatch(r"(\d+),(\d+):(\d+),(\d+)", line)
        if match:
            result_digits.append(int(match.group(3)))
            final_carry = int(match.group(4))
    if final_carry:
        result_digits.append(final_carry)
    raw = sum(digit * 10**index for index, digit in enumerate(result_digits))
    return raw * 10**shift


def verify_annotation(row: dict[str, str], annotation: dict[str, object] | str) -> None:
    """Independently replay a rendered scratchpad against its source row.

    Raises ``ValueError`` at the first malformed or mathematically inconsistent
    definition. This validates emitted local digit transitions, reference
    equations, division remainder/GCD steps, final answers, and algebra
    substitution. It accepts either an annotated row or scratchpad text.
    """
    scratchpad = (
        str(annotation["scratchpad"])
        if isinstance(annotation, dict)
        else annotation
    )
    if isinstance(annotation, dict) and annotation.get("trace_version") == "reasoning_v2":
        _verify_forward_annotation(row, scratchpad)
        return
    module = row.get("module", "")
    answer = _fraction(row["answer"])
    if module == "algebra__linear_1d":
        _, _, source_value = _linear_trace(row["question"], row["answer"])
    else:
        source_expression, _ = _source_expression(row["question"], module)
        source_expression = re.sub(r"(?<=\d)\s+(?=[+-]?\d)", " ", source_expression)
        source_value = _ExpressionParser(source_expression, _Builder()).parse().value
    if source_value != answer:
        raise ValueError("Source-row arithmetic does not produce its labeled answer")
    source_operands = None
    if module in {"arithmetic__add_or_sub", "arithmetic__mul", "arithmetic__div"}:
        source_expression, _ = _source_expression(row["question"], module)
        simple = re.fullmatch(
            rf"\s*({NUMBER})\s*([+*/-])\s*({NUMBER})\s*", source_expression
        )
        if simple:
            source_operands = (_fraction(simple.group(1)), simple.group(2), _fraction(simple.group(3)))
    definitions: dict[str, list[str]] = {}
    for line in scratchpad.splitlines():
        match = _DEFINITION.fullmatch(line)
        if match:
            name, detail = match.groups()
            if name in definitions:
                raise ValueError(f"Duplicate scratchpad definition {name}")
            definitions[name] = [detail]
        elif not definitions:
            raise ValueError(f"Scratchpad line precedes its first definition: {line!r}")
        else:
            definitions[next(reversed(definitions))].append(line)

    values: dict[str, Fraction] = {}

    def operand(text: str) -> Fraction:
        text = text.strip()
        reference = re.fullmatch(r"([+-]?)([tp]\d+)", text)
        if reference:
            sign, name = reference.groups()
            if name not in values:
                raise ValueError(f"Reference {name} is missing or used before definition")
            return -values[name] if sign == "-" else values[name]
        return _fraction(text)

    for name, detail_lines in definitions.items():
        lines = detail_lines
        detail = "\n".join(lines)
        head = lines[0]
        if name.startswith("p"):
            partial_collapse = re.fullmatch(r"(-?1)\*(\d+)\*(\d+)\*10\^(\d+)=(-?\d+)", head)
            if partial_collapse:
                sign, multiplicand, digit, shift_text, value_text = partial_collapse.groups()
                shift = int(shift_text)
                if source_operands and source_operands[1] == "*":
                    source_multiplicand, _, source_multiplier = source_operands
                    source_scale = _scale(source_multiplicand)
                    if abs(int(source_multiplicand * 10**source_scale)) != int(multiplicand):
                        raise ValueError("Collapsed partial product multiplicand disagrees with source")
                    multiplier_integer = abs(int(source_multiplier * 10**_scale(source_multiplier)))
                    source_digits = _integer_digits(multiplier_integer)
                    expected_digit = source_digits[-1 - shift] if shift < len(source_digits) else 0
                    if int(digit) != expected_digit:
                        raise ValueError("Collapsed partial-product digit disagrees with source multiplier")
                result = int(multiplicand) * int(digit) * 10**int(shift_text)
                result *= int(sign)
                if result != int(value_text):
                    raise ValueError(f"Collapsed partial product {name} is incorrect")
                values[name] = Fraction(result)
                continue
            multiplication_header = next(
                (line for line in lines if line.startswith("M") and "@" in line), None
            )
            if multiplication_header is None:
                raise ValueError(f"Partial product {name} has no multiplier/shift header")
            match = re.fullmatch(r"M(.+?)@(\d+)=(?:sign:)?(-?\d+)", multiplication_header)
            if not match:
                raise ValueError(f"Malformed partial-product header: {multiplication_header!r}")
            multiplier_text, shift_text, value_text = match.groups()
            shift = int(shift_text)
            source_multiplicand = None
            if source_operands and source_operands[1] == "*":
                source_multiplicand, _, source_multiplier = source_operands
                if _fraction(multiplier_text) != source_multiplier:
                    raise ValueError("Partial-product multiplier disagrees with the source row")
            expected = _verify_multiplier_rows(
                _fraction(multiplier_text), shift,
                lines[lines.index(multiplication_header) + 1 :], source_multiplicand
            )
            signed = "sign:" in multiplication_header
            if signed:
                expected = -expected
            if expected != int(value_text):
                raise ValueError(f"Partial product {name} disagrees with its digit rows")
            values[name] = Fraction(expected)
            continue

        sum_header = re.fullmatch(
            r"([+-]?[tp]\d+)\+([+-]?[tp]\d+)(?:=| ->)(-?\d+(?:/\d+)?)?", head
        )
        if sum_header:
            left, right, result_text = sum_header.groups()
            a, b = operand(left), operand(right)
            result = a + b
            if len(lines) > 1:
                if a.denominator != 1 or b.denominator != 1:
                    raise ValueError("Digit addition references must resolve to integers")
                replayed = _verify_addition_rows(int(a), int(b), lines[1:])
                if Fraction(replayed) != result:
                    raise ValueError(f"Referenced addition {name} disagrees with its digit rows")
            elif result_text and _fraction(result_text) != result:
                raise ValueError(f"Referenced addition {name} has an incorrect result")
            values[name] = result
            continue

        header_index = 1 if head == "sign:-" else 0
        operation_head = lines[header_index]
        binary = re.fullmatch(r"([+*/-])\(([^,]+),([^)]*)\)=([^\s]+)", operation_head)
        if binary:
            symbol, left, right, result_text = binary.groups()
            a, b, stated = operand(left), operand(right), _fraction(result_text)
            if source_operands and name == "t0":
                source_a, source_symbol, source_b = source_operands
                if (symbol, a, b) != (source_symbol, source_a, source_b):
                    raise ValueError("First operation operands disagree with the source row")
            if symbol == "+":
                expected = a + b
            elif symbol == "-":
                expected = a - b
            elif symbol == "*":
                expected = a * b
            else:
                if b == 0:
                    raise ValueError("Division trace has a zero divisor")
                expected = a / b
            if stated != expected:
                raise ValueError(f"Arithmetic result in {name} is incorrect")
            operation_rows = lines[header_index + 1 :]
            if symbol in "+-" and operation_rows:
                operation_rows = _verify_decimal_scale(a, b, expected, operation_rows)
                scale = max(_scale(a), _scale(b))
                ai, bi = int(a * 10**scale), int(b * 10**scale)
                replayed = (
                    _verify_addition_rows(ai, bi if symbol == "+" else -bi, operation_rows)
                )
                if Fraction(replayed, 10**scale) != expected:
                    raise ValueError(f"Digit arithmetic in {name} disagrees with its equation")
            if symbol == "/" and operation_rows:
                _verify_division_detail(a, b, expected, operation_rows)
            values[name] = expected
            continue

        atom = r"[+-]?(?:[tp]\d+|\d+(?:/\d+)?)"
        equation = re.fullmatch(rf"({atom})([+*/-])({atom})=(-?\d+(?:/\d+)?)", head)
        if equation:
            left, symbol, right, result_text = equation.groups()
            a, b, stated = operand(left), operand(right), _fraction(result_text)
            expected = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[symbol]
            if expected is None or stated != expected:
                raise ValueError(f"Referenced operation {name} is incorrect")
            values[name] = expected
            if symbol in "+-" and len(lines) > 1:
                if a.denominator != 1 or b.denominator != 1:
                    raise ValueError("Referenced digit addition must use integer values")
                detail_rows = _verify_decimal_scale(a, b, expected, lines[1:])
                replayed = _verify_addition_rows(
                    int(a), int(b) if symbol == "+" else -int(b), detail_rows
                )
                if Fraction(replayed) != expected:
                    raise ValueError(f"Referenced digit transitions in {name} are incorrect")
            if symbol == "/" and len(lines) > 1:
                _verify_division_detail(a, b, expected, lines[1:])
            continue

        if head.startswith("scale:"):
            scale_match = re.fullmatch(r"scale:(-?\d+)/(\d+)=([^\s]+)", head)
            if not scale_match:
                raise ValueError(f"Malformed multiplication scale line: {head!r}")
            numerator, denominator, rendered = scale_match.groups()
            expected = Fraction(int(numerator), int(denominator))
            if _fraction(rendered) != expected:
                raise ValueError(f"Scale result in {name} is incorrect")
            prior_values = list(values.values())
            if not prior_values or prior_values[-1] != int(numerator):
                raise ValueError("Multiplication scale numerator disagrees with prior result")
            values[name] = expected
            continue
        collapsed_scale = re.fullmatch(r"([tp]\d+)/(\d+)=([^\s]+)", head)
        if collapsed_scale:
            previous, denominator_text, answer_text = collapsed_scale.groups()
            expected = operand(previous) / int(denominator_text)
            if expected != _fraction(answer_text):
                raise ValueError("Collapsed multiplication scale reference is incorrect")
            values[name] = expected
            continue

        # Algebra is rendered as one essential node with explicit source
        # substitution. Its arithmetic subexpressions are checked below.
        if head.startswith("collect:"):
            _verify_algebra(row, lines)
            solution = re.search(r"(?m)^([a-z])=([^\n]+)$", detail)
            if not solution:
                raise ValueError("Algebra trace has no solution line")
            values[name] = _fraction(solution.group(2))
            continue
        raise ValueError(f"Unrecognized mathematical definition {name}: {head!r}")

    answer = _fraction(row["answer"])
    if not values or list(values.values())[-1] != answer:
        raise ValueError("Final trace value does not match the source answer")


def _verify_division_detail(a: Fraction, b: Fraction, result: Fraction, rows: list[str]) -> None:
    common = math.lcm(a.denominator, b.denominator)
    dividend = abs(a.numerator * (common // a.denominator))
    divisor = abs(b.numerator * (common // b.denominator))
    transitions = rows
    gcd_line = next((line for line in transitions if line.startswith("gcd=")), None)
    digit_lines = [line for line in transitions if not line.startswith("sign:") and line != gcd_line]
    if divisor == 0:
        raise ValueError("Division trace has zero divisor")
    remainder = 0
    if len(digit_lines) != len(str(dividend)):
        raise ValueError("Division trace has the wrong number of brought-down digits")
    for digit, line in zip(str(dividend), digit_lines):
        brought = remainder * 10 + int(digit)
        quotient_digit, remainder = divmod(brought, divisor)
        if line != f"{brought}:{quotient_digit},{remainder}":
            raise ValueError("Long-division quotient/remainder transition does not replay")
    if gcd_line is None:
        raise ValueError("Division trace has no GCD reduction")
    match = re.fullmatch(r"gcd=(\d+);(-?\d+)/(\d+)", gcd_line)
    if not match:
        raise ValueError("Malformed division GCD line")
    stated_gcd, numerator, denominator = map(int, match.groups())
    actual_gcd = _gcd(dividend, divisor)
    sign = -1 if a * b < 0 else 1
    if (stated_gcd != actual_gcd or Fraction(numerator, denominator) != result
            or numerator != sign * dividend // actual_gcd or denominator != divisor // actual_gcd):
        raise ValueError("Division GCD reduction does not match the source operands")


def _verify_algebra(row: dict[str, str], lines: list[str]) -> None:
    equation, variable = _source_expression(row["question"], "algebra__linear_1d")
    left, right = equation.split("=", 1)
    solution_line = next((line for line in lines if line.startswith(f"{variable}=")), None)
    verification = next((line for line in lines if line.startswith("verify:")), None)
    if solution_line is None or verification is None:
        raise ValueError("Algebra trace lacks substitution verification")
    solution = _fraction(solution_line.split("=", 1)[1])
    if solution != _fraction(row["answer"]):
        raise ValueError("Algebra solution disagrees with source answer")
    left_value = _linear_value(left, variable, solution)
    right_value = _linear_value(right, variable, solution)
    stated = verification.removeprefix("verify:").split("=", 1)
    if len(stated) != 2 or _fraction(stated[0]) != left_value or _fraction(stated[1]) != right_value:
        raise ValueError("Algebra substitution values do not replay from the source equation")
    if left_value != right_value:
        raise ValueError("Algebra solution does not satisfy the source equation")


def _linear_value(text: str, variable: str, value: Fraction) -> Fraction:
    tokens = re.findall(r"\d+(?:\.\d*)?|\.\d+|[a-z]|[()+*/-]", text)
    if "".join(tokens) != re.sub(r"\s+", "", text):
        raise ValueError("Unsupported syntax in algebra verification")
    replaced = "".join(_scalar(value) if token == variable else token for token in tokens)
    expression = _ExpressionParser(replaced, _Builder()).parse()
    return expression.value


def annotation_hash(rows: Iterable[dict[str, object]]) -> str:
    """Stable digest for annotated rows/provenance (canonical JSONL encoding)."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def distribution(values: list[int]) -> dict[str, int | float]:
    """Compact length statistics used by the Hydra annotation entrypoint."""
    if not values:
        return {"count": 0, "min": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {"count": len(ordered), "min": ordered[0], "mean": statistics.mean(ordered),
            "median": statistics.median(ordered), "p95": ordered[int(.95 * (len(ordered)-1))],
            "p99": ordered[int(.99 * (len(ordered)-1))], "max": ordered[-1]}
