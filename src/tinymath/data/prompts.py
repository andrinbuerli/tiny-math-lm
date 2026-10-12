"""Single prompt/completion rendering shared by preparation, training and evaluation."""

QUESTION_PREFIX = "Q: "
ANSWER_PREFIX = "\nA: "


def format_prompt(question: str) -> str:
    return f"{QUESTION_PREFIX}{question}{ANSWER_PREFIX}"


def format_completion(answer: str) -> str:
    return answer
