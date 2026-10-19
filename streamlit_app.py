"""Streamlit interface for interactive TinyMath checkpoint inference."""

import html
import re
import time
from pathlib import Path

import streamlit as st
import torch

from tinymath.data.trace_explanations import transcribe_scratchpad
from tinymath.inference import (
    discover_inference_checkpoints,
    format_raw_text,
    format_tool_call,
    load_inference_bundle,
    sample_split_example,
    stream_inference_events,
    unsupported_characters,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = "runs/exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f-eval1k"

st.set_page_config(page_title="TinyMath Inference", page_icon="∑", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] .main .block-container {
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        height: 100dvh;
        min-height: 0;
    }
    [data-testid="stAppViewContainer"] .main .block-container > div[data-testid="stVerticalBlock"] {
        flex: 1 1 auto;
        min-height: 0;
    }
    .st-key-chat_history {
        flex: 1 1 0%;
        min-height: 0;
    }
    .tm-reasoning-activity {
        align-items: center;
        color: var(--text-color, #737373);
        display: flex;
        font-size: 0.92rem;
        gap: 0.55rem;
        line-height: 1.45;
        opacity: 0.78;
        padding: 0.25rem 0;
    }
    .tm-reasoning-activity span {
        overflow-wrap: anywhere;
        white-space: pre-wrap;
    }
    .tm-reasoning-activity.is-active::before {
        animation: tm-reasoning-pulse 1.6s ease-out infinite;
        background: #7799d3;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(119, 153, 211, 0.42);
        content: "";
        flex: 0 0 0.48rem;
        height: 0.48rem;
    }
    .tm-reasoning-activity.is-complete::before {
        color: #5f9b76;
        content: "✓";
        flex: 0 0 0.48rem;
        font-size: 0.82rem;
        font-weight: 700;
        text-align: center;
    }
    @keyframes tm-reasoning-pulse {
        0% { box-shadow: 0 0 0 0 rgba(119, 153, 211, 0.42); opacity: 0.85; }
        70% { box-shadow: 0 0 0 0.42rem rgba(119, 153, 211, 0); opacity: 1; }
        100% { box-shadow: 0 0 0 0 rgba(119, 153, 211, 0); opacity: 0.85; }
    }
    @media (prefers-reduced-motion: reduce) {
        .tm-reasoning-activity.is-active::before { animation: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading checkpoint…")
def cached_bundle(checkpoint: str, device: str, modified: float):
    del modified
    return load_inference_bundle(checkpoint, repository_root=ROOT, device=device)


def _tool_markdown(tool: dict) -> str:
    rendered = [format_tool_call(tool["operation"], tool["left"], tool["right"])]
    if tool.get("task"):
        rendered.extend(["", f"*{format_raw_text(tool['task'])}*"])
    if tool.get("sidecar"):
        sidecar = format_raw_text(tool["sidecar"].rstrip("\r\n"))
        quoted = "\n".join(
            f"> {line}" if line else ">"
            for line in sidecar.splitlines()
        )
        rendered.extend(["", quoted])
    if tool.get("sidecar_error"):
        rendered.extend(["", f"> Recursive LM error: {format_raw_text(tool['sidecar_error'])}"])
    if tool.get("error"):
        rendered.extend(["", f"> ⚠ {format_raw_text(tool['error'])}"])
    return "\n".join(rendered)


def _recursive_result_markdown(tool: dict) -> str:
    if tool.get("result") is None:
        return ""
    return f"→ Recursive LM: {format_raw_text(tool['result'])}"


def _render_saved_row(row: dict) -> None:
    left_column, right_column = st.columns([3, 2])
    left, right = _row_markdown(row)
    left_column.markdown(left)
    right_column.markdown(right)
    tool = row.get("tool")
    if tool and tool.get("result") is not None:
        result_left, result_right = st.columns([3, 2])
        result_left.markdown(_recursive_result_markdown(tool))
        result_right.markdown("Computed by the Recursive LM.")


def _row_markdown(row: dict) -> tuple[str, str]:
    left = format_raw_text(row["left"])
    if row.get("tool") is not None:
        left = "\n\n".join(section for section in (left, _tool_markdown(row["tool"])) if section)
    left = "\n\n".join(
        section for section in (left, *(f"> ⚠ {format_raw_text(item)}" for item in row["notices"]))
        if section
    )
    if row["left"].lstrip().startswith("A:"):
        return "", ""
    return left, format_raw_text(row["explanation"])


def _final_answer(response: str) -> str:
    answers = re.findall(r"(?m)^A:\s*(.*)$", response)
    return answers[-1].strip() if answers else ""


def _reasoning_activity(line: str, *, active: bool) -> str:
    state = "is-active" if active else "is-complete"
    text = html.escape(line) if line else "Thinking…"
    return (
        f'<div class="tm-reasoning-activity {state}" role="status">'
        f"<span>{text}</span></div>"
    )


def _saved_row(row: dict) -> dict:
    return {
        "left": row["left"],
        "tool": dict(row["tool"]) if row["tool"] is not None else None,
        "explanation": row["explanation"],
        "notices": list(row["notices"]),
    }


def _render_reference(reference: dict) -> None:
    with st.expander("Reference solution", expanded=False):
        st.markdown(f"**Answer:** {format_raw_text(reference.get('answer', 'Unavailable'))}")
    with st.expander("Reference scratchpad", expanded=False):
        scratchpad = reference.get("scratchpad")
        st.code(scratchpad if scratchpad else "No reference scratchpad in this dataset row.")


with st.sidebar:
    st.header("Checkpoint")
    run_path = st.text_input("Run directory", value=DEFAULT_RUN)
    run_root = (ROOT / run_path).resolve()
    checkpoints = discover_inference_checkpoints(run_root)
    if checkpoints:
        checkpoint_parts = {item.label: Path(item.label).parts for item in checkpoints}
        model_sizes = sorted(
            {parts[0] for parts in checkpoint_parts.values() if len(parts) >= 3},
            key=lambda value: int(value.removesuffix("m")),
        )
        default_size = "1m" if "1m" in model_sizes else model_sizes[0]
        model_size = st.selectbox(
            "Model size", model_sizes, index=model_sizes.index(default_size), key="model_size"
        )
        size_checkpoints = [
            item for item in checkpoints if checkpoint_parts[item.label][0] == model_size
        ]
        arms = sorted({checkpoint_parts[item.label][1] for item in size_checkpoints})
        if st.session_state.get("training_arm") not in arms:
            st.session_state.training_arm = arms[0]
        training_arm = st.selectbox("Training arm", arms, key="training_arm")
        arm_checkpoints = [
            item for item in size_checkpoints
            if checkpoint_parts[item.label][1] == training_arm
        ]
        seeds = sorted({checkpoint_parts[item.label][2] for item in arm_checkpoints})
        if st.session_state.get("checkpoint_seed") not in seeds:
            st.session_state.checkpoint_seed = seeds[0]
        checkpoint_seed = st.selectbox("Seed", seeds, key="checkpoint_seed")
        selected = next(
            item for item in arm_checkpoints
            if checkpoint_parts[item.label][2] == checkpoint_seed
        )
        st.caption(f"Selected checkpoint: {selected.label}")
    else:
        selected = None
        st.warning(f"No latest_inference.pt snapshots found under {run_root}")
    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device = st.selectbox("Device", device_options)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_prompt" not in st.session_state:
    st.session_state.chat_prompt = ""
if "sampled_reference" not in st.session_state:
    st.session_state.sampled_reference = None

if selected is not None:
    try:
        model, tokenizer, metadata = cached_bundle(
            str(selected.path), device, selected.path.stat().st_mtime
        )

        with st.sidebar:
            st.divider()
            st.header("Inference")
            st.subheader("Question")
            source = st.selectbox(
                "Question source",
                ["Freeform", "Sample from train", "Sample from validation", "Sample from test"],
            )
            if source != "Freeform" and st.button("Sample question"):
                split = source.removeprefix("Sample from ")
                try:
                    example = sample_split_example(
                        split, data_bundle=metadata["data_bundle"]
                    )
                    st.session_state.chat_prompt = example["question"]
                    st.session_state.sampled_reference = {
                        "split": split,
                        "question": example["question"],
                        "answer": example.get("answer", "Unavailable"),
                        "scratchpad": example.get("scratchpad", ""),
                    }
                except (OSError, ValueError, TypeError) as exc:
                    st.session_state.sampled_reference = None
                    st.error(f"Could not sample that split: {exc}")

            st.subheader("Generation")
            max_tokens = st.slider(
                "Max output chars (both streams)", 8, min(512, model.config.context_length),
                min(256, model.config.context_length), 8,
            )
            temperature = st.slider("Temperature", 0.0, 1.5, 0.0, 0.05)
            st.divider()
            if st.button("Clear chat"):
                st.session_state.chat_history.clear()
                st.session_state.sampled_reference = None
                st.session_state.chat_prompt = ""

        prompt = st.chat_input("Ask a math question", key="chat_prompt")
        sampled_reference = st.session_state.sampled_reference
        chat_history_container = st.container(
            height="stretch", border=False, key="chat_history", autoscroll=True
        )
        with chat_history_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown(message["content"])
                        if message.get("reference"):
                            _render_reference(message["reference"])
                        continue
                    if message.get("latest_reasoning_line"):
                        st.markdown(
                            _reasoning_activity(message["latest_reasoning_line"], active=False),
                            unsafe_allow_html=True,
                        )
                    if message.get("rows"):
                        with st.status("Reasoning", state="complete", expanded=False):
                            for saved in message["rows"]:
                                _render_saved_row(saved)
                    if message.get("answer"):
                        st.markdown(message["answer"])

            if prompt is None and sampled_reference is not None:
                st.caption(f"Sampled from {sampled_reference['split']}; reference is UI-only.")
                _render_reference(sampled_reference)
        if prompt is not None:
            unsupported = unsupported_characters(tokenizer, prompt)
            if unsupported:
                rendered = ", ".join(
                    f"{character!r} (U+{ord(character):04X})" for character in unsupported
                )
                st.warning(f"This checkpoint cannot encode: {rendered}")
            else:
                with chat_history_container:
                    reference = (
                        sampled_reference
                        if sampled_reference is not None and sampled_reference["question"] == prompt
                        else None
                    )
                    st.session_state.sampled_reference = None
                    st.session_state.chat_history.append({
                        "role": "user", "content": prompt, "reference": reference,
                    })
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        if reference is not None:
                            _render_reference(reference)
                    assistant_history = {
                        "role": "assistant", "rows": [], "answer": "",
                        "latest_reasoning_line": "",
                    }
                    st.session_state.chat_history.append(assistant_history)
                    with st.chat_message("assistant"):
                        stats_view = st.empty()
                        latest_step_view = st.empty()
                        generation_state = {"active": True}
                        reasoning_status = st.status("Reasoning", expanded=False)
                        answer_view = st.empty()
                        with reasoning_status:
                            response_rows = st.container()
                            generation_started = time.perf_counter()
                            calls = {}
                            stats = {"input": 0, "agent_output": 0, "subagent": 0}
                            parser = {
                                "scratchpad_lines": [], "scratchpad_active": False,
                                "scratchpad_line": "", "outside_line": "",
                                "known_explanation_count": 0,
                            }
                            row_state = {"current": None, "rows": []}
                            generated_response = ""
                            latest_line_state = {"text": ""}

                            def render_latest_step(line=""):
                                if line.strip():
                                    latest_line_state["text"] = line
                                if not latest_line_state["text"]:
                                    latest_step_view.empty()
                                    return
                                latest_step_view.markdown(
                                    _reasoning_activity(
                                        latest_line_state["text"], active=generation_state["active"]
                                    ),
                                    unsafe_allow_html=True,
                                )

                            def new_row():
                                with response_rows:
                                    left_column, right_column = st.columns([3, 2])
                                    result_left_column, result_right_column = st.columns([3, 2])
                                    row = {
                                        "left": "", "tool": None, "explanation": "", "notices": [],
                                        "left_view": left_column.empty(),
                                        "right_view": right_column.empty(),
                                        "result_left_view": result_left_column.empty(),
                                        "result_right_view": result_right_column.empty(),
                                    }
                                row_state["rows"].append(row)
                                row_state["current"] = row
                                return row

                            def render_row(row):
                                left, right = _row_markdown(row)
                                row["left_view"].markdown(left)
                                row["right_view"].markdown(right)
                                if row["tool"]:
                                    result = _recursive_result_markdown(row["tool"])
                                    row["result_left_view"].markdown(result)
                                    row["result_right_view"].markdown(
                                        "Computed by the Recursive LM."
                                        if row["tool"].get("result") is not None else ""
                                    )

                            def explain_line(line, *, is_scratchpad):
                                if not line.strip() or line in {"<scp>", "</scp>"}:
                                    return
                                if is_scratchpad:
                                    render_latest_step(line)
                                    parser["scratchpad_lines"].append(line)
                                    explanation = None
                                    try:
                                        explanations = transcribe_scratchpad(
                                            "\n".join(parser["scratchpad_lines"]),
                                            version="reasoning_v3",
                                        )
                                        if len(explanations) > parser["known_explanation_count"]:
                                            explanation = explanations[-1]
                                            parser["known_explanation_count"] = len(explanations)
                                    except ValueError:
                                        pass
                                    if explanation is None:
                                        explanation = f"Trace line: {line}."
                                elif line.startswith("A:"):
                                    explanation = f"This is the model's final answer: {line[2:].strip()}."
                                else:
                                    explanation = f"Model output: {line}."
                                row_state["current"]["explanation"] = explanation
                                render_row(row_state["current"])

                            def consume_prediction(text, *, display=True):
                                row = row_state["current"]
                                for character in text:
                                    if display:
                                        row["left"] += character
                                    if parser["scratchpad_active"]:
                                        if character == "\n":
                                            if parser["scratchpad_line"] == "</scp>":
                                                parser["scratchpad_active"] = False
                                            else:
                                                explain_line(parser["scratchpad_line"], is_scratchpad=True)
                                            parser["scratchpad_line"] = ""
                                        else:
                                            parser["scratchpad_line"] += character
                                    else:
                                        parser["outside_line"] += character
                                        if parser["outside_line"].endswith("<scp>\n"):
                                            parser["outside_line"] = ""
                                            parser["scratchpad_active"] = True
                                        elif character == "\n":
                                            line = parser["outside_line"][:-1].rstrip("\r")
                                            parser["outside_line"] = ""
                                            explain_line(line, is_scratchpad=False)
                                    if display:
                                        render_row(row)
                                    if display and character == "\n":
                                        row = new_row()

                            def render_stats():
                                elapsed = max(time.perf_counter() - generation_started, 1e-9)
                                output = stats["agent_output"] + stats["subagent"]
                                total = stats["input"] + output
                                with stats_view.container():
                                    columns = st.columns(6)
                                    values = (
                                        total,
                                        stats["input"],
                                        output,
                                        stats["agent_output"],
                                        stats["subagent"],
                                        f"{total / elapsed:.1f}",
                                    )
                                    labels = (
                                        "Total tokens",
                                        "Input tokens (prompt)",
                                        "Output tokens",
                                        "Agent tokens",
                                        "Subagent tokens",
                                        "Tokens / s (overall)",
                                    )
                                    for column, label, value in zip(columns, labels, values):
                                        column.metric(label, value)

                            render_stats()
                            new_row()
                            try:
                                for event in stream_inference_events(
                                    model, tokenizer, prompt,
                                    max_new_tokens=max_tokens,
                                    sidecar_max_new_tokens=max_tokens,
                                    temperature=temperature,
                                ):
                                    kind = event["kind"]
                                    if kind == "generation_start":
                                        stats["input"] += event["input_tokens"]
                                    elif kind == "primary_token":
                                        tool_result = event.get("source") == "tool"
                                        if not tool_result:
                                            generated_response += event["text"]
                                        consume_prediction(event["text"], display=not tool_result)
                                        if tool_result:
                                            answer = _final_answer(generated_response)
                                            if answer:
                                                answer_view.markdown(format_raw_text(answer))
                                            continue
                                        stats["agent_output"] += 1
                                        answer = _final_answer(generated_response)
                                        if answer:
                                            answer_view.markdown(format_raw_text(answer))
                                    elif kind == "tool_started":
                                        row = row_state["current"]
                                        request = event["request"]
                                        row["left"] = row["left"].removesuffix(request)
                                        row["tool"] = {
                                            "operation": event["operation"], "left": event["left"],
                                            "right": event["right"], "sidecar": "",
                                        }
                                        calls[event["call_id"]] = row
                                        render_row(row)
                                    elif kind == "sidecar_start":
                                        stats["subagent"] += event["input_tokens"]
                                        row = calls[event["call_id"]]
                                        row["tool"]["task"] = event["task"]
                                        render_row(row)
                                    elif kind == "sidecar_token":
                                        row = calls[event["call_id"]]
                                        row["tool"]["sidecar"] += event["text"]
                                        stats["subagent"] += 1
                                        render_row(row)
                                    elif kind == "tool_result":
                                        row = calls[event["call_id"]]
                                        row["tool"]["result"] = event["result"]
                                        render_row(row)
                                    elif kind in {"tool_error", "sidecar_error"}:
                                        row = calls.get(event.get("call_id"), row_state["current"])
                                        key = "sidecar_error" if kind == "sidecar_error" else "error"
                                        if row["tool"] is not None:
                                            row["tool"][key] = event["text"]
                                        else:
                                            row["notices"].append(f"Tool request not executed: {event['text']}")
                                        render_row(row)
                                    elif kind == "generation_error":
                                        row_state["current"]["notices"].append(event["text"])
                                        render_row(row_state["current"])
                                    render_stats()
                            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                                row_state["current"]["notices"].append(str(exc))
                                render_row(row_state["current"])

                            if parser["scratchpad_line"] and parser["scratchpad_line"] != "</scp>":
                                explain_line(parser["scratchpad_line"], is_scratchpad=parser["scratchpad_active"])
                            if parser["outside_line"].strip():
                                explain_line(parser["outside_line"].strip(), is_scratchpad=False)
                            generation_state["active"] = False
                            render_latest_step()
                            answer = _final_answer(generated_response)
                            if not answer:
                                answer = "No final answer generated."
                            answer_view.markdown(format_raw_text(answer))
                            reasoning_status.update(label="Reasoning", state="complete", expanded=False)
                            assistant_history["answer"] = answer
                            assistant_history["latest_reasoning_line"] = latest_line_state["text"]
                            assistant_history["rows"] = [
                                _saved_row(row) for row in row_state["rows"]
                                if row["left"].strip() or row["tool"] or row["notices"]
                            ]
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        st.error(f"Could not load this checkpoint: {exc}")
