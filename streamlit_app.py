"""Streamlit interface for interactive TinyMath checkpoint inference."""

from pathlib import Path

import streamlit as st
import torch

from tinymath.inference import (
    discover_inference_checkpoints,
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
        selected_label = st.selectbox("Model / training arm", [item.label for item in checkpoints])
        selected = next(item for item in checkpoints if item.label == selected_label)
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
            "Max output chars", 8, min(512, model.config.context_length),
            min(256, model.config.context_length), 8,
        )
        temperature = control_columns[1].slider("Temperature", 0.0, 1.5, 0.0, 0.05)
        run = control_columns[3].button("Generate", type="primary", disabled=bool(unsupported))

        primary_column, sidecar_column = st.columns(2)
        primary_column.subheader("Checkpoint response")
        sidecar_column.subheader("Parallel model response to tool calls")
        primary_view = primary_column.empty()
        sidecar_view = sidecar_column.empty()
        tool_view = st.container()

        if run:
            primary_text = ""
            sidecar_text = ""
            with st.spinner("Generating streamed output…"):
                for event in stream_inference_events(
                    model,
                    tokenizer,
                    question,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if event["kind"] == "primary_token":
                        primary_text += event["text"]
                        primary_view.code(primary_text or " ", language=None)
                    elif event["kind"] == "sidecar_token":
                        sidecar_text += event["text"]
                        sidecar_view.code(sidecar_text or " ", language=None)
                    elif event["kind"] == "tool_started":
                        tool_view.info(f"Exact tool running: {event['request'].strip()}")
                    elif event["kind"] == "tool_result":
                        tool_view.success(
                            f"Exact tool result: {event['request'].strip()}{event['result']}"
                        )
                    elif event["kind"] == "tool_error":
                        tool_view.warning(f"Tool request not executed: {event['text']}")
                    elif event["kind"] in {"generation_error", "sidecar_error"}:
                        st.error(event["text"])
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        st.error(f"Could not load this checkpoint: {exc}")

st.divider()
st.caption("Start with `uv run streamlit run streamlit_app.py` from the repository root.")
