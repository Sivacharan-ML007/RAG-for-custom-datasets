from __future__ import annotations

import streamlit as st

from rag.config import CONFIG
from rag.qa import answer_question, format_sources
from rag.spreadsheets import SUPPORTED_DATASET_TYPES
from rag.vectorstore import build_vectorstore


st.set_page_config(
    page_title="Hugging Face Dataset RAG",
    page_icon=":material/table_chart:",
    layout="wide",
)


with st.sidebar:
    st.header("Hugging Face RAG")

    uploaded_files = st.file_uploader(
        "Upload dataset files",
        type=sorted(
            extension.removeprefix(".") for extension in SUPPORTED_DATASET_TYPES
        ),
        accept_multiple_files=True,
    )

    hf_model_id = st.text_input(
        "Hugging Face model",
        value=CONFIG.hf_model_id,
        placeholder="Qwen/Qwen2.5-7B-Instruct",
    )
    hf_token = st.text_input(
        "Hugging Face token",
        value=CONFIG.hf_token,
        type="password",
        placeholder="hf_...",
    )
    embedding_model = st.text_input("Embedding model", value=CONFIG.embedding_model)

    with st.expander("Retrieval settings"):
        top_k = st.slider("Retrieved chunks", 1, 12, CONFIG.top_k)
        chunk_size = st.slider("Chunk size", 300, 2000, CONFIG.chunk_size, step=100)
        chunk_overlap = st.slider(
            "Chunk overlap", 0, 400, CONFIG.chunk_overlap, step=20
        )

    with st.expander("Generation settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)
        max_new_tokens = st.slider("Max answer tokens", 128, 2048, 768, step=128)

    build_clicked = st.button("Build index", type="primary", use_container_width=True)


st.title("Dataset RAG")

files_column, ask_column = st.columns([0.38, 0.62], gap="large")

with files_column:
    st.subheader("Files")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")
    else:
        st.info(
            "Upload CSV, Excel, JSON, Parquet, Feather, ODS, or delimited text files "
            "from the sidebar."
        )

    if "chunk_count" in st.session_state:
        st.success(f"Index ready with {st.session_state.chunk_count} chunks.")


with ask_column:
    st.subheader("Ask")

    if build_clicked:
        if not uploaded_files:
            st.error("Upload at least one dataset file first.")
        else:
            with st.spinner("Reading datasets and building the local vector index..."):
                try:
                    vectorstore, chunk_count = build_vectorstore(
                        uploaded_files,
                        embedding_model,
                        chunk_size,
                        chunk_overlap,
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunk_count = chunk_count
                    st.success(f"Index built with {chunk_count} chunks.")
                except Exception as exc:
                    st.error(str(exc))

    question = st.text_area(
        "Question",
        placeholder="Ask about totals, rows, names, dates, notes, or patterns in the uploaded datasets.",
        height=120,
    )

    ask_clicked = st.button("Ask", use_container_width=True)

    if ask_clicked:
        if "vectorstore" not in st.session_state:
            st.error("Build the index before asking a question.")
        elif not hf_model_id.strip():
            st.error("Enter a Hugging Face model id.")
        elif not hf_token.strip():
            st.error("Enter your Hugging Face access token.")
        elif not question.strip():
            st.error("Enter a question first.")
        else:
            with st.spinner(
                "Retrieving rows and generating the answer with Hugging Face..."
            ):
                try:
                    answer = answer_question(
                        vectorstore=st.session_state.vectorstore,
                        question=question,
                        model_id=hf_model_id,
                        hf_token=hf_token,
                        top_k=top_k,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.markdown(answer["result"])

                    sources = format_sources(answer)
                    if sources:
                        st.caption("Sources: " + " | ".join(sources))
