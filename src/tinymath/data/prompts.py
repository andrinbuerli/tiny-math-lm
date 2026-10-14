"""Single prompt/completion rendering shared by preparation, training and evaluation."""

QUESTION_PREFIX = "Q: "
ANSWER_PREFIX = "\nA: "


def format_prompt(question: str, *, protocol: str = "answer_only") -> str:
    if protocol == "answer_only":
        return f"{QUESTION_PREFIX}{question}{ANSWER_PREFIX}"
    if protocol == "reasoning_v1":
        return f"{QUESTION_PREFIX}{question}\n"
    raise ValueError(f"Unsupported completion protocol: {protocol}")


def format_completion(answer: str) -> str:
    return answer


def format_reasoning_display(scratchpad: str, answer: str) -> str:
    """Return the readable completion form; token IDs are inserted by the tokenizer."""
    if not scratchpad.strip() or not answer.strip():
        raise ValueError("Reasoning completions require a nonempty scratchpad and answer")
    return f"<scp>\n{scratchpad}\n</scp>\nA: {answer}"
