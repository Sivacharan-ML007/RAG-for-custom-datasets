from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    hf_model_id: str = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    hf_token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))


CONFIG = AppConfig()
