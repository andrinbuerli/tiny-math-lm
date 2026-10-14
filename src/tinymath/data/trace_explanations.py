"""Turn verified compact reasoning traces into readable step explanations."""

from __future__ import annotations

import re

_DEFINITION = re.compile(r"^([tp]\d+)=(.*)$")
_ADD_ROW = re.compile(r"^(\d+),(\d+),(\d+):(\d+),(\d+)$")
_MUL_ROW = re.compile(r"^(\d+),(\d+):(\d+),(\d+)$")
_DIV_ROW = re.compile(r"^(\d+):(\d+),(\d+)$")


def _display_operand(value: str, known: dict[str, str]) -> str:
    match = re.fullmatch(r"([+-]?)([tp]\d+)", value.strip())
    if match:
        sign, name = match.groups()
        reference = f"step {int(name[1:]) + 1}"
        value = known.get(name)
        if value:
            reference += f" ({value})"
        if sign == "-":
            return (
                f"the opposite of {reference}"
                if value and value.startswith("-")
                else f"negative {reference}"
            )
        return reference
    return value.strip()


def _is_negative(value: str, known: dict[str, str]) -> bool:
    match = re.fullmatch(r"([+-]?)([tp]\d+)", value.strip())
    if not match:
        return value.strip().startswith("-")
    sign, name = match.groups()
    is_negative = known.get(name, "").startswith("-")
    return not is_negative if sign == "-" else is_negative


def _arithmetic_step(line: str, known: dict[str, str]) -> tuple[str, str] | None:
    match = re.fullmatch(r"([+*/-])\((.*?),(.*?)\)=([+-]?[\d./]+)", line)
    if not match:
        return None
    symbol, left, right, result = match.groups()
    left_text = _display_operand(left, known)
    right_text = _display_operand(right, known)
    expression = f"{left_text} {symbol} {right_text} = {result}"
    action = {
        "+": f"Add {left_text} and {right_text}",
        "-": f"Subtract {right_text} from {left_text}",
        "*": f"Multiply {left_text} by {right_text}",
        "/": f"Divide {left_text} by {right_text}",
    }[symbol]
    return f"{action}: {expression}.", result


def _explain_group(name: str, lines: list[str], known: dict[str, str]) -> list[str]:
    if not lines:
        return []
    head = lines[0]
    explanation: list[str] = []

    if head.startswith("collect:"):
        equation = head.removeprefix("collect:")
        explanation.append(f"Collect the variable and constant terms on each side: {equation}.")
        coefficient = None
        for line in lines[1:]:
            if line.startswith("move:"):
                move = re.fullmatch(r"move:(-?\d+)[a-z]=(-?[\d./]+)", line)
                if move:
                    coefficient = move.group(1)
                explanation.extend(_explain_algebra_line(line))
            elif assignment := re.fullmatch(r"([a-z])=(-?[\d./]+)", line):
                variable, solution = assignment.groups()
                if coefficient:
                    explanation.append(
                        f"Divide both sides by {coefficient} to isolate {variable}: "
                        f"{variable} = {solution}."
                    )
                else:
                    explanation.append(f"The solution is {variable} = {solution}.")
                known[name] = solution
            else:
                explanation.extend(_explain_algebra_line(line))
        return explanation

    if head.startswith("scale:"):
        content = head.removeprefix("scale:")
        if "=>" in content:
            places, result = content.split("=>", 1)
            explanation.append(f"Restore the decimal point by scaling by 10^{places}: {result}.")
            known[name] = result
        elif "=" in content:
            fraction, result = content.split("=", 1)
            explanation.append(f"Apply the decimal scale ({fraction}) to get {result}.")
            known[name] = result
        return explanation

    collapsed_partial = re.fullmatch(r"(-?1)\*(\d+)\*(\d+)\*10\^(\d+)=(-?\d+)", head)
    if collapsed_partial:
        sign, multiplicand, digit, shift, result = collapsed_partial.groups()
        sign_text = "Apply the negative sign and " if sign.startswith("-") else ""
        explanation.append(
            f"{sign_text}multiply {multiplicand} by the multiplier digit {digit}, "
            f"then shift {shift} place(s): {result}."
        )
        known[name] = result
        return explanation

    multiplication_header = re.fullmatch(r"M(.+?)@(\d+)=(?:sign:)?(-?\d+)", head)
    if multiplication_header:
        multiplier, shift_text, result = multiplication_header.groups()
        shift = int(shift_text)
        multiplier_digits = multiplier.lstrip("+-")
        digit = multiplier_digits[-1 - shift] if shift < len(multiplier_digits) else "0"
        digit_rows = [match.groups() for line in lines[1:] if (match := _MUL_ROW.fullmatch(line))]
        multiplicand = "".join(row[0] for row in digit_rows)[::-1] or "the current number"
        sign_text = "Apply the negative sign and " if result.startswith("-") else ""
        explanation.append(
            f"{sign_text}multiply {multiplicand} by the {digit} in the {shift + 1}"
            f"'s place of {multiplier}; this partial product is {result}."
        )
        for row in lines[1:]:
            if row.startswith("digit:"):
                digits = row.removeprefix("digit:").replace(",", " ")
                explanation.append(f"Read the multiplier digits from right to left: {digits}.")
            elif row.startswith("carry:"):
                explanation.append(f"Carry {row.split(':', 1)[1]} into the next place.")
            elif match := _MUL_ROW.fullmatch(row):
                digit_value, carry_in, out, carry_out = match.groups()
                explanation.append(
                    f"Multiply digit {digit_value} by {digit}, add carry {carry_in}, "
                    f"write {out}, and carry {carry_out}."
                )
        known[name] = result
        return explanation

    division_head = next(
        (line for line in lines if re.fullmatch(r"/\((.*?),(.*?)\)=([+-]?[\d./]+)", line)),
        None,
    )
    if "sign:-" in lines:
        explanation.append("The operands have opposite signs, so the quotient is negative.")
    division_header = (
        re.fullmatch(r"/\((.*?),(.*?)\)=([+-]?[\d./]+)", division_head) if division_head else None
    )
    if division_header:
        dividend, divisor, result = division_header.groups()
        explanation.append(f"Divide {dividend} by {divisor}: the quotient is {result}.")
        if dividend.startswith("-") and divisor.startswith("-"):
            explanation.append("The two negative signs cancel, so the quotient is positive.")
        for row in lines:
            if row == division_head or row == "sign:-":
                continue
            if match := _DIV_ROW.fullmatch(row):
                brought, quotient, remainder = match.groups()
                explanation.append(
                    f"Long division: use {brought}, write quotient digit {quotient}, "
                    f"and carry remainder {remainder}."
                )
            elif row.startswith("gcd="):
                gcd, fraction = row.removeprefix("gcd=").split(";", 1)
                explanation.append(
                    f"Reduce the fraction by the greatest common divisor {gcd}: {fraction}."
                )
        known[name] = result
        return explanation

    arithmetic = _arithmetic_step(head, known)
    if arithmetic:
        narrative, result = arithmetic
        explanation.append(narrative)
        known[name] = result
        operation = re.fullmatch(r"([+*/-])\((.*?),(.*?)\)=([+-]?[\d./]+)", head)
        subtraction_rows = bool(
            operation
            and operation.group(1) == "-"
            and _is_negative(operation.group(2), known) == _is_negative(operation.group(3), known)
        )
        for row in lines[1:]:
            if match := _ADD_ROW.fullmatch(row):
                left, right, carry_in, digit, carry_out = match.groups()
                if subtraction_rows:
                    explanation.append(
                        f"Subtract digit {right} and borrow {carry_in} from {left}: "
                        f"write {digit} and pass borrow {carry_out} to the next place."
                    )
                else:
                    explanation.append(
                        f"Add the current digits {left} and {right} with carry {carry_in}: "
                        f"write {digit} and carry {carry_out} to the next place."
                    )
            elif row.startswith("carry:"):
                explanation.append(f"Write the final carry {row.split(':', 1)[1]}.")
            elif match := _DIV_ROW.fullmatch(row):
                brought, quotient, remainder = match.groups()
                explanation.append(
                    f"Long division: use {brought}, write quotient digit {quotient}, "
                    f"and carry remainder {remainder}."
                )
            elif row.startswith("gcd="):
                gcd, fraction = row.removeprefix("gcd=").split(";", 1)
                explanation.append(
                    f"Reduce the fraction by the greatest common divisor {gcd}: {fraction}."
                )
            elif row.startswith("scale:"):
                explanation.extend(_explain_group(name, [row], known))
        return explanation

    if match := re.fullmatch(r"([tp]\d+)\+([tp]\d+)=(-?\d+)", head):
        left, right, result = match.groups()
        explanation.append(
            f"Add the partial results from {_display_operand(left, known)} and "
            f"{_display_operand(right, known)} to get {result}."
        )
        known[name] = result
        for row in lines[1:]:
            if match := _ADD_ROW.fullmatch(row):
                left_digit, right_digit, carry_in, digit, carry_out = match.groups()
                explanation.append(
                    f"Add digits {left_digit} and {right_digit} with carry {carry_in}: "
                    f"write {digit}, carry {carry_out}."
                )
        return explanation

    if head.startswith("move:"):
        explanation.extend(_explain_algebra_line(head))
        known[name] = head.removeprefix("move:").split("=", 1)[-1]
        return explanation

    if head.startswith("verify:"):
        explanation.extend(_explain_algebra_line(head))
        return explanation

    if match := re.fullmatch(r"([a-z])=(-?[\d./]+)", head):
        variable, solution = match.groups()
        move = next(
            (re.fullmatch(r"move:(-?\d+)[a-z]=(-?[\d./]+)", line) for line in lines[1:]), None
        )
        coefficient = move.group(1) if move else None
        if coefficient:
            explanation.append(
                f"Divide both sides by {coefficient} to isolate {variable}: {variable} = {solution}."
            )
        else:
            explanation.append(f"The solution is {variable} = {solution}.")
        known[name] = solution
        return explanation

    if match := re.fullmatch(r"([+-]?[\d./]+)([+*/-])([+-]?[\d./]+)=(-?[\d./]+)", head):
        left, symbol, right, result = match.groups()
        arithmetic = _arithmetic_step(f"{symbol}({left},{right})={result}", known)
        if arithmetic:
            explanation.append(arithmetic[0])
            known[name] = result
            return explanation

    return [f"Trace step {name}: {head}."]


def _explain_algebra_line(line: str) -> list[str]:
    if line.startswith("move:"):
        match = re.fullmatch(r"move:(-?\d+)([a-z])=(-?[\d./]+)", line)
        if match:
            coefficient, variable, constant = match.groups()
            return [
                (
                    f"Move the variable terms to the left and constants to the right: "
                    f"{coefficient}{variable} = {constant}."
                )
            ]
        return [f"Rearrange the equation: {line.removeprefix('move:')}."]
    if line.startswith("verify:"):
        left, _, right = line.removeprefix("verify:").partition("=")
        return [
            f"Check the solution in the original equation: both sides equal {left} (right side {right})."
        ]
    return []


def explain_reasoning_trace(scratchpad: str) -> list[str]:
    """Explain each operation in a verified compact trace using plain language."""
    groups: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for line in scratchpad.splitlines():
        match = _DEFINITION.fullmatch(line)
        if match:
            if current_name is not None:
                groups.append((current_name, current_lines))
            current_name, first_line = match.groups()
            current_lines = [first_line]
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        groups.append((current_name, current_lines))

    known: dict[str, str] = {}
    explanations: list[str] = []
    for name, lines in groups:
        explanations.extend(_explain_group(name, lines, known))
    return explanations


TRACE_KEY = {
    "t": "A temporary result from an arithmetic or algebra step.",
    "p": "A partial product in long multiplication.",
    "a,b,c:d,e": "For addition, input digits a and b plus carry-in c produce digit d and carry-out e. For subtraction, c and e are the borrow-in and borrow-out.",
    "a,b:c,d": "For multiplication, digit a times the current multiplier digit, plus carry-in b, produces digit c and carry-out d.",
    "a:q,r": "A long-division step: the current chunk a produces quotient digit q and remainder r.",
    "scale": "Restore a decimal point after doing integer arithmetic on scaled values.",
    "gcd": "Reduce a fraction by its greatest common divisor.",
    "collect / move / verify": "For equations: combine like terms, isolate the variable, then check both sides.",
}
