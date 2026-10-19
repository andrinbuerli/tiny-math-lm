"""Streamlit interface for interactive TinyMath checkpoint inference."""

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


@st.cache_resource(show_spinner="Loading checkpoint…")
def cached_bundle(checkpoint: str, device: str, modified: float):
    del modified
    return load_inference_bundle(checkpoint, repository_root=ROOT, device=device)


def _tool_markdown(tool: dict) -> str:
    rendered = [format_tool_call(tool["operation"], tool["left"], tool["right"])]
    if tool.get("task"):
        rendered.extend(["", f"_{format_raw_text(tool['task'])}_"])
    if tool.get("sidecar"):
        quoted = "\n".join(
            f"> {line}" if line else ">"
            for line in format_raw_text(tool["sidecar"]).splitlines()
        )
        rendered.extend(["", quoted])
    if tool.get("sidecar_error"):
        rendered.extend(["", f"> Parallel LM error: {format_raw_text(tool['sidecar_error'])}"])
    if tool.get("result") is not None:
        rendered.extend(["", f"→ Parallel LM: {format_raw_text(tool['result'])}"])
    if tool.get("error"):
        rendered.extend(["", f"> ⚠ {format_raw_text(tool['error'])}"])
    return "\n".join(rendered)


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
        params = sum(parameter.numel() for parameter in model.parameters())
        cols = st.columns(4)
        cols[0].metric("Parameters", f"{params / 1_000_000:.1f}M")
        cols[1].metric("Protocol", metadata["protocol"])
        cols[2].metric("Training arm", metadata["training_arm"])
        cols[3].metric("Update", metadata["update"] if metadata["update"] is not None else "—")

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

        st.caption("Each turn is independent. Earlier messages are displayed here but are not sent to the model.")
        prompt = st.chat_input("Ask a math question", key="chat_prompt")
        sampled_reference = st.session_state.sampled_reference
        chat_history_container = st.container(height=480, border=False)
        with chat_history_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown(message["content"])
                        if message.get("reference"):
                            _render_reference(message["reference"])
                        continue
                    if message.get("rows"):
                        with st.status("Reasoning", state="complete", expanded=False):
                            for saved in message["rows"]:
                                left_column, right_column = st.columns([3, 2])
                                left, right = _row_markdown(saved)
                                left_column.markdown(left)
                                right_column.markdown(right)
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
                    assistant_history = {"role": "assistant", "rows": [], "answer": ""}
                    st.session_state.chat_history.append(assistant_history)
                    with st.chat_message("assistant"):
                        stats_view = st.empty()
                        reasoning_status = st.status("Reasoning", expanded=True)
                        answer_view = st.empty()
                        with reasoning_status:
                            response_rows = st.container()
                            generation_started = time.perf_counter()
                            calls = {}
                            stats = {"input": 0, "output": 0, "reasoning": 0}
                            parser = {
                                "scratchpad_lines": [], "scratchpad_active": False,
                                "scratchpad_line": "", "outside_line": "",
                                "known_explanation_count": 0,
                            }
                            row_state = {"current": None, "rows": []}
                            generated_response = ""

                            def new_row():
                                with response_rows:
                                    left_column, right_column = st.columns([3, 2])
                                    row = {
                                        "left": "", "tool": None, "explanation": "", "notices": [],
                                        "left_view": left_column.empty(),
                                        "right_view": right_column.empty(),
                                    }
                                row_state["rows"].append(row)
                                row_state["current"] = row
                                return row

                            def render_row(row):
                                left, right = _row_markdown(row)
                                row["left_view"].markdown(left)
                                row["right_view"].markdown(right)

                            def explain_line(line, *, is_scratchpad):
                                if not line.strip() or line in {"<scp>", "</scp>"}:
                                    return
                                if is_scratchpad:
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
                                total = stats["input"] + stats["output"]
                                with stats_view.container():
                                    columns = st.columns(5)
                                    values = (total, stats["input"], stats["output"],
                                              stats["reasoning"], f"{stats['output'] / elapsed:.1f}")
                                    labels = ("Total tokens", "Input tokens", "Output tokens",
                                              "Reasoning tokens", "Tokens / s")
                                    for column, label, value in zip(columns, labels, values):
                                        column.metric(label, value)

                            render_stats()
                            new_row()
                            try:
                                with st.spinner("Generating streamed output…"):
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
                                            stats["output"] += 1
                                            if event.get("region") == "reasoning":
                                                stats["reasoning"] += 1
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
                                            stats["reasoning"] = max(0, stats["reasoning"] - event["tool_tokens"])
                                            render_row(row)
                                        elif kind == "sidecar_start":
                                            stats["input"] += event["input_tokens"]
                                            row = calls[event["call_id"]]
                                            row["tool"]["task"] = event["task"]
                                            render_row(row)
                                        elif kind == "sidecar_token":
                                            row = calls[event["call_id"]]
                                            row["tool"]["sidecar"] += event["text"]
                                            stats["output"] += 1
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
                            answer = _final_answer(generated_response)
                            if not answer:
                                answer = "No final answer generated."
                            answer_view.markdown(format_raw_text(answer))
                            reasoning_status.update(label="Reasoning", state="complete", expanded=False)
                            assistant_history["answer"] = answer
                            assistant_history["rows"] = [
                                _saved_row(row) for row in row_state["rows"]
                                if row["left"].strip() or row["tool"] or row["notices"]
                            ]
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        st.error(f"Could not load this checkpoint: {exc}")

st.divider()
st.caption("Start with `uv run streamlit run streamlit_app.py` from the repository root.")
