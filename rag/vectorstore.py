from __future__ import annotations

from typing import Iterable

import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.spreadsheets import documents_from_upload


@st.cache_resource(show_spinner=False)
def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(
    uploaded_files,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[FAISS, int]:
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    documents: list[Document] = []
    for uploaded_file in uploaded_files:
        documents.extend(documents_from_upload(uploaded_file))

    if not documents:
        raise ValueError("No usable rows were found in the uploaded files.")

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No searchable text was found after splitting the dataset rows.")

    embeddings = load_embeddings(embedding_model)
    try:
        return FAISS.from_documents(chunks, embeddings), len(chunks)
    except Exception as exc:
        raise RuntimeError(
            "Could not build embeddings. Check your internet connection and embedding model name."
        ) from exc


def split_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(list(documents))
