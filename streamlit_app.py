"""Streamlit interface for interactive TinyMath checkpoint inference."""

import time
from pathlib import Path

import streamlit as st
import torch

from tinymath.inference import (
    discover_inference_checkpoints,
    format_raw_text,
    format_tool_call,
    load_inference_bundle,
    sample_split_question,
    stream_inference_events,
    unsupported_characters,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = "runs/exp13_reasoning_v3_arithmetic_subtasks-fresh-5d493a0f-eval1k"

st.set_page_config(page_title="TinyMath Inference", page_icon="∑", layout="wide")
st.title("TinyMath checkpoint playground")
st.caption("Ask a new question or sample one from this experiment's data splits.")


@st.cache_resource(show_spinner="Loading checkpoint…")
def cached_bundle(checkpoint: str, device: str, modified: float):
    del modified
    return load_inference_bundle(checkpoint, repository_root=ROOT, device=device)


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

if "question" not in st.session_state:
    st.session_state.question = "Calculate 23 + 48."

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

        source_column, sample_column = st.columns([3, 1])
        source = source_column.selectbox(
            "Question source", ["Freeform", "Sample from train", "Sample from validation", "Sample from test"]
        )
        if source != "Freeform" and sample_column.button("Sample question"):
            split = source.removeprefix("Sample from ")
            try:
                st.session_state.question = sample_split_question(
                    split, data_bundle=metadata["data_bundle"]
                )
            except (OSError, ValueError, TypeError) as exc:
                st.error(f"Could not sample that split: {exc}")

        question = st.text_area(
            "Math question",
            height=100,
            key="question",
            placeholder="Enter a question, or choose a split and sample one.",
        )
        unsupported = unsupported_characters(tokenizer, question)
        if unsupported:
            rendered = ", ".join(f"{character!r} (U+{ord(character):04X})" for character in unsupported)
            st.warning(f"This checkpoint cannot encode: {rendered}")

        control_columns = st.columns([1, 1, 2, 1])
        max_tokens = control_columns[0].slider(
            "Max output chars (both streams)", 8, min(512, model.config.context_length),
            min(256, model.config.context_length), 8,
        )
        temperature = control_columns[1].slider("Temperature", 0.0, 1.5, 0.0, 0.05)
        run = control_columns[3].button("Generate", type="primary", disabled=bool(unsupported))

        stats_view = st.empty()
        st.subheader("Streamed response")
        response_view = st.empty()

        if run:
            generation_started = time.perf_counter()
            parts = [{"kind": "model", "text": ""}]
            calls = {}
            stats = {"input": 0, "output": 0, "reasoning": 0}
            notices = []

            def render_call(part):
                rendered = [format_tool_call(part["operation"], part["left"], part["right"])]
                if part.get("task"):
                    rendered.extend(["", f"_{format_raw_text(part['task'])}_"])
                if part.get("sidecar"):
                    quoted = "\n".join(
                        f"> {line}" if line else ">"
                        for line in format_raw_text(part["sidecar"]).splitlines()
                    )
                    rendered.extend(["", quoted])
                if part.get("sidecar_error"):
                    rendered.extend(["", f"> Parallel LM error: {format_raw_text(part['sidecar_error'])}"])
                if part.get("result") is not None:
                    rendered.extend(["", f"**→ `{part['result']}`**"])
                if part.get("error"):
                    rendered.extend(["", f"> ⚠ {format_raw_text(part['error'])}"])
                return "\n".join(rendered)

            def render_response():
                sections = [
                    format_raw_text(part["text"]) if part["kind"] == "model" else render_call(part)
                    for part in parts
                ]
                sections.extend(f"> ⚠ {format_raw_text(notice)}" for notice in notices)
                response_view.markdown("\n\n".join(section for section in sections if section))

            def render_stats():
                stats["total"] = stats["input"] + stats["output"]
                elapsed = max(time.perf_counter() - generation_started, 1e-9)
                with stats_view.container():
                    columns = st.columns(5)
                    for column, label, key in zip(
                        columns,
                        ("Total tokens", "Input tokens", "Output tokens", "Reasoning tokens", "Tokens / s"),
                        ("total", "input", "output", "reasoning", "speed"),
                    ):
                        value = f"{stats['output'] / elapsed:.1f}" if key == "speed" else stats[key]
                        column.metric(label, value)

            with st.spinner("Generating streamed output…"):
                for event in stream_inference_events(
                    model,
                    tokenizer,
                    question,
                    max_new_tokens=max_tokens,
                    sidecar_max_new_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if event["kind"] == "generation_start":
                        stats["input"] += event["input_tokens"]
                        render_stats()
                    elif event["kind"] == "primary_token":
                        if event.get("source") == "tool":
                            continue
                        parts[-1]["text"] += event["text"]
                        stats["output"] += 1
                        if event.get("region") == "reasoning":
                            stats["reasoning"] += 1
                        render_response()
                        render_stats()
                    elif event["kind"] == "tool_started":
                        current = parts[-1]["text"]
                        request = event["request"]
                        if current.endswith(request):
                            parts[-1]["text"] = current[:-len(request)]
                        part = {
                            "kind": "tool",
                            "operation": event["operation"],
                            "left": event["left"],
                            "right": event["right"],
                            "sidecar": "",
                        }
                        parts.append(part)
                        parts.append({"kind": "model", "text": ""})
                        calls[event["call_id"]] = part
                        stats["reasoning"] = max(
                            0, stats["reasoning"] - event["tool_tokens"]
                        )
                        render_response()
                        render_stats()
                    elif event["kind"] == "sidecar_start":
                        stats["input"] += event["input_tokens"]
                        calls[event["call_id"]]["task"] = event["task"]
                        render_response()
                        render_stats()
                    elif event["kind"] == "sidecar_token":
                        calls[event["call_id"]]["sidecar"] += event["text"]
                        stats["output"] += 1
                        render_response()
                        render_stats()
                    elif event["kind"] == "tool_result":
                        calls[event["call_id"]]["result"] = event["result"]
                        render_response()
                    elif event["kind"] == "tool_error":
                        if event.get("call_id") in calls:
                            calls[event["call_id"]]["error"] = event["text"]
                        else:
                            notices.append(f"Tool request not executed: {event['text']}")
                        render_response()
                    elif event["kind"] == "sidecar_error":
                        calls[event["call_id"]]["sidecar_error"] = event["text"]
                        render_response()
                    elif event["kind"] == "generation_error":
                        notices.append(event["text"])
                        render_response()
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        st.error(f"Could not load this checkpoint: {exc}")

st.divider()
st.caption("Start with `uv run streamlit run streamlit_app.py` from the repository root.")
