"""Streamlit workbench for asking, corpus inspection, and evaluation.

Usage:
    streamlit run app.py
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from support_copilot.config import AppConfig, configure_llamaindex
from support_copilot.evaluation.runner import run_answers, run_retrieval
from support_copilot.ingestion.document_loader import SUPPORTED_EXTENSIONS, discover_files
from support_copilot.ingestion.document_register import load_document_register
from support_copilot.ingestion.processed_records import load_records
from support_copilot.services.query_service import QueryService


def main() -> None:
    """Render the complete quality workbench."""

    base_config = AppConfig.from_env()
    st.set_page_config(page_title="Support Copilot", layout="wide")
    st.title("Support Copilot")

    with st.sidebar:
        provider = st.selectbox("Provider", ["ollama", "anthropic"], index=0)
        mode = st.segmented_control("Retriever", ["semantic", "bm25", "hybrid"], default="hybrid")
        top_k = st.slider("Top-K", 1, 10, base_config.top_k)
        st.code(str(base_config.raw_data_dir), language=None)

    ask_tab, corpus_tab, evaluation_tab = st.tabs(
        ["Ask Assistant", "Knowledge Base Health", "Evaluation Dashboard"]
    )

    with ask_tab:
        with st.form("ask_form"):
            question = st.text_input("Question", value="Which power supply is recommended for the Raspberry Pi 5?")
            submitted = st.form_submit_button("Ask", type="primary")
        if submitted and not question.strip():
            st.warning("Enter a question first.")
        if submitted and question.strip():
            try:
                config = replace(base_config, llm_provider=provider)
                configure_llamaindex(config)
                result = QueryService(config).answer(question, top_k=top_k, mode=mode or "hybrid")
                st.subheader("Answer")
                st.write(result.answer)
                if result.unverified_claims:
                    st.warning(
                        "No retrieved source supports the sentences marked [Unverified]; "
                        "check them against the evidence below."
                    )
                with st.expander("Raw model answer (before code-attached citations)"):
                    st.text(result.model_answer)
                st.subheader("Sources")
                st.dataframe(pd.DataFrame([item.__dict__ for item in result.citations]), hide_index=True)
                with st.expander("Retrieved evidence", expanded=True):
                    for item in result.evidence:
                        metadata = item.node.metadata
                        st.markdown(
                            f"**{metadata.get('source', 'unknown')} | "
                            f"{metadata.get('location', 'unknown')} | score {item.score:.4f}**"
                        )
                        if mode == "hybrid":
                            st.caption(
                                f"Semantic rank: {metadata.get('semantic_rank')} | "
                                f"BM25 rank: {metadata.get('bm25_rank')}"
                            )
                        st.write(item.node.get_content())
            except Exception as exc:
                st.error(str(exc))

    with corpus_tab:
        files = discover_files(base_config.raw_data_dir)
        columns = st.columns(3)
        columns[0].metric("Raw files", len(files))
        columns[1].metric("Supported extensions", len(SUPPORTED_EXTENSIONS))
        records = load_records(base_config.processed_records_path) if base_config.processed_records_path.exists() else []
        columns[2].metric("Prepared records", len(records))
        st.write("Supported:", ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        if files:
            st.dataframe(
                pd.DataFrame(
                    {
                        "source": path.relative_to(base_config.raw_data_dir).as_posix(),
                        "extension": path.suffix.lower(),
                    }
                    for path in files
                ),
                hide_index=True,
            )
        report_path = base_config.reports_dir / "corpus_preparation.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            st.subheader("Preparation status")
            st.dataframe(pd.DataFrame(report["files"]), hide_index=True)
        if records:
            st.subheader("Normalized evidence preview")
            st.dataframe(
                pd.DataFrame(
                    {
                        "source": row["metadata"].get("source"),
                        "location": row["metadata"].get("location"),
                        "type": row["metadata"].get("document_type"),
                        "method": row["metadata"].get("extraction_method"),
                        "text": row["text"][:220],
                    }
                    for row in records[:100]
                ),
                hide_index=True,
            )
        register = load_document_register(base_config.document_register_path)
        if register:
            st.subheader("Document register")
            st.dataframe(pd.DataFrame(register), hide_index=True)

    with evaluation_tab:
        run_name = st.text_input("Run name", value="streamlit-baseline")
        if st.button("Run retrieval evaluation"):
            try:
                configure_llamaindex(base_config)
                _, summary_path = run_retrieval(
                    base_config, mode=mode or "hybrid", top_k=top_k, run_name=run_name
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                st.bar_chart(pd.DataFrame([summary])[['hit_at_k', 'recall_at_k', 'context_precision', 'mrr']].T)
                st.json(summary)
            except Exception as exc:
                st.error(str(exc))
        confirm = st.checkbox("Run the selected LLM once for every evaluation question")
        if st.button("Run answer evaluation", disabled=not confirm):
            try:
                config = replace(base_config, llm_provider=provider)
                configure_llamaindex(config)
                paths = run_answers(config, mode=mode or "hybrid", top_k=top_k, run_name=run_name)
                st.success("Answer and manual-rubric reports created.")
                for path in paths:
                    st.download_button(path.name, path.read_bytes(), file_name=path.name)
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
