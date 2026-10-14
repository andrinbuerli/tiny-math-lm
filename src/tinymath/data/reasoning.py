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
            factor = 10**scale
            ai, bi, ri = int(a * factor), int(b * factor), int(result * factor)
            detail = _add_lines(ai, bi, ri) if symbol == "+" else _add_lines(ai, -bi, ri)
            expression = f"{symbol}({_decimal(a)},{_decimal(b)})={_decimal(result)}"
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
) -> dict[str, str | int | dict[str, int]]:
    """Verify and annotate a source row without mutating it."""
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
    full_length = length_fn(row, full_text)
    scratch, collapsed, retained_length = reduce_to_budget(
        row, operations, max_tokens=max_tokens, length_fn=length_fn)
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
    return annotated


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
