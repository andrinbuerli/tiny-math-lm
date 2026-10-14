"""Local Streamlit browser for prepared TinyMath data bundles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from tinymath.data.browser import discover_bundles, read_jsonl

SPLIT_LABELS = {
    "sft_train": "SFT train",
    "sft_validation": "SFT validation",
    "rlvr": "RLVR prompts",
    "test": "Interpolation test",
    "number_length_ood": "Number length OOD",
    "expression_length_ood": "Expression length OOD",
    "extrapolate_other": "Other extrapolation",
}
MODULE_LABELS = {
    "arithmetic__add_or_sub": "Addition / subtraction",
    "arithmetic__mul": "Multiplication",
    "arithmetic__div": "Division",
    "arithmetic__mixed": "Mixed arithmetic",
    "algebra__linear_1d": "Linear equations",
}


@st.cache_data(show_spinner="Loading dataset split…")
def cached_jsonl(path: str, modified_ns: int) -> list[dict[str, str]]:
    """Cache a split until its on-disk modification time changes."""
    del modified_ns
    return read_jsonl(Path(path))


def _read_manifest(bundle: Path) -> dict[str, Any]:
    with (bundle / "manifest.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    st.set_page_config(page_title="TinyMath data browser", page_icon="∑", layout="wide")
    st.title("TinyMath dataset browser")
    st.caption("Explore the prepared, leakage-controlled data pools used by TinyMath-RL.")

    repo_root = Path(__file__).resolve().parents[1]
    candidates = discover_bundles(repo_root / "data")
    if not candidates:
        st.info("No prepared data bundles found.")
        st.code("uv run python scripts/prepare_data.py experiment=01_smoke", language="bash")
        st.caption(f"Looking for bundles in `{repo_root / 'data' / 'processed'}`.")
        return

    bundle_by_name = {f"{path.name}": path for path in candidates}
    with st.sidebar:
        st.header("Browse")
        selected_bundle = st.selectbox("Prepared bundle", list(bundle_by_name))
    bundle = bundle_by_name[selected_bundle]
    try:
        manifest = _read_manifest(bundle)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read `{bundle / 'manifest.json'}`: {exc}")
        return

    files = manifest.get("splits", {})
    available = [name for name in SPLIT_LABELS if name in files]
    if not available:
        st.error("This bundle manifest does not list any recognized data splits.")
        return

    selected_split = st.sidebar.selectbox(
        "Data pool", available, format_func=lambda value: SPLIT_LABELS[value]
    )
    split_info = files[selected_split]
    split_path = bundle / str(split_info.get("path", f"{selected_split}.jsonl"))
    if not split_path.is_file():
        st.error(f"Split file is missing: `{split_path}`")
        return
    try:
        rows = cached_jsonl(str(split_path), split_path.stat().st_mtime_ns)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not load this split: {exc}")
        return

    modules = sorted({row.get("module", "") for row in rows if row.get("module")})
    categories = sorted({row.get("category", "") for row in rows if row.get("category")})
    module_counts = Counter(row.get("module", "unknown") for row in rows)
    total = len(rows)

    st.subheader(SPLIT_LABELS[selected_split])
    metric_cols = st.columns(4)
    metric_cols[0].metric("Examples in pool", f"{total:,}")
    metric_cols[1].metric("Modules", len(modules))
    metric_cols[2].metric("Dataset revision", str(manifest.get("dataset_revision", "unknown"))[:12])
    metric_cols[3].metric("Context length", manifest.get("context_length", "—"))

    with st.expander("Bundle provenance", expanded=False):
        provenance_cols = st.columns(3)
        provenance_cols[0].write(f"**Dataset:** {manifest.get('dataset_id', 'unknown')}")
        provenance_cols[1].write(
            f"**Manifest hash:** `{manifest.get('split_manifest_sha256', 'unknown')}`"
        )
        provenance_cols[2].write(f"**Seed:** {manifest.get('seed', 'unknown')}")
        st.json(
            {
                "source_split": manifest.get("dataset_split"),
                "selected_modules": manifest.get("selected_modules", []),
                "split_rules": manifest.get("split_rules", {}),
                "tokenizer": manifest.get("tokenizer", {}),
            }
        )

    left, right = st.columns([1, 2])
    with left:
        st.markdown("#### Module coverage")
        coverage = pd.DataFrame(
            [
                {"Module": MODULE_LABELS.get(module, module), "Examples": count}
                for module, count in sorted(module_counts.items())
            ]
        )
        if not coverage.empty:
            st.bar_chart(coverage.set_index("Module"), width="stretch")
            st.dataframe(coverage, hide_index=True, width="stretch")

    with right:
        st.markdown("#### Examples")
        filter_cols = st.columns([1, 1, 2])
        chosen_module = filter_cols[0].selectbox(
            "Module",
            ["All modules", *modules],
            format_func=lambda value: MODULE_LABELS.get(value, value),
        )
        chosen_category = filter_cols[1].selectbox("Category", ["All categories", *categories])
        search = filter_cols[2].text_input("Search questions or answers")

        filtered = [
            row
            for row in rows
            if (chosen_module == "All modules" or row.get("module") == chosen_module)
            and (chosen_category == "All categories" or row.get("category") == chosen_category)
            and (
                not search
                or search.casefold()
                in f"{row.get('question', '')} {row.get('answer', '')}".casefold()
            )
        ]
        st.caption(f"Showing {len(filtered):,} of {total:,} examples")
        page_size = st.select_slider("Rows per page", options=[10, 25, 50, 100], value=25)
        pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1)
        start = (page - 1) * page_size
        page_rows = filtered[start : start + page_size]
        display_rows = [
            {
                "Module": MODULE_LABELS.get(row.get("module", ""), row.get("module", "")),
                "Category": row.get("category", ""),
                "Question": row.get("question", ""),
                "Answer": row.get("answer", ""),
            }
            for row in page_rows
        ]
        st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")
        if page_rows:
            st.markdown("#### Selected example")
            selected_index = st.selectbox(
                "Inspect row on this page",
                range(len(page_rows)),
                format_func=lambda index: (
                    f"{start + index + 1}: {page_rows[index].get('question', '')[:90]}"
                ),
            )
            selected = page_rows[selected_index]
            st.markdown(f"**Question**\n\n{selected.get('question', '')}")
            st.markdown(f"**Answer**\n\n{selected.get('answer', '')}")
            st.caption(f"{selected.get('module', '')} · {selected.get('category', '')}")


if __name__ == "__main__":
    main()
