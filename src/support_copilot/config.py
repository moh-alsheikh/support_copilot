"""Central configuration for corpus preparation, indexing, retrieval, generation, and evaluation.

All paths are anchored to the project root, so commands work from the project directory and
the Streamlit process sees the same files. Environment variables are loaded before values are read.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def _env_bool(name: str, default: bool) -> bool:
    """Read common true/false spellings from an environment variable."""

    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Immutable settings shared by scripts, services, tests, and Streamlit."""

    root_dir: Path
    raw_data_dir: Path
    processed_records_path: Path
    document_register_path: Path
    readiness_path: Path
    evaluation_dataset_path: Path
    reports_dir: Path
    storage_dir: Path
    llm_provider: str
    llm_model: str
    embedding_model: str
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    collection_name: str = "technical_evidence"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # nomic-embed-text is trained with these task prefixes; other embedding models may need none.
    embedding_query_prefix: str = "search_query:"
    embedding_document_prefix: str = "search_document:"
    request_timeout: float = 360.0
    top_k: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 64
    ocr_enabled: bool = True
    ocr_min_chars: int = 40
    ocr_dpi: int = 300
    ocr_language: str = "eng"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load `.env`, then construct absolute project paths and model choices."""

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        return cls(
            root_dir=root,
            raw_data_dir=root / "data" / "raw",
            processed_records_path=root / "data" / "processed" / "document_records.jsonl",
            document_register_path=root / "data" / "metadata" / "document_register.csv",
            readiness_path=root / "data" / "metadata" / "project_readiness.csv",
            evaluation_dataset_path=root / "evaluation" / "datasets" / "evaluation_questions.json",
            reports_dir=root / "evaluation" / "reports",
            storage_dir=root / "storage",
            llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
            llm_model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            collection_name=os.getenv("COLLECTION_NAME", "technical_evidence"),
            reranker_model=os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2").strip(),
            embedding_query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", "search_query:").strip(),
            embedding_document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", "search_document:").strip(),
            request_timeout=float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "360")),
            top_k=int(os.getenv("TOP_K", "4")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            ocr_enabled=_env_bool("OCR_ENABLED", True),
            ocr_min_chars=int(os.getenv("OCR_MIN_CHARS", "40")),
            ocr_dpi=int(os.getenv("OCR_DPI", "300")),
            ocr_language=os.getenv("OCR_LANGUAGE", "eng"),
        )


def configure_llamaindex(config: AppConfig) -> None:
    """Register the selected generator and local embedding model with LlamaIndex."""

    if config.llm_provider == "ollama":
        Settings.llm = Ollama(model=config.llm_model, request_timeout=config.request_timeout)
    elif config.llm_provider == "anthropic":
        if not config.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
        from llama_index.llms.anthropic import Anthropic

        Settings.llm = Anthropic(model=config.anthropic_model, api_key=config.anthropic_api_key)
    else:
        raise ValueError("LLM_PROVIDER must be 'ollama' or 'anthropic'.")
    Settings.embed_model = OllamaEmbedding(
        model_name=config.embedding_model,
        query_instruction=config.embedding_query_prefix or None,
        text_instruction=config.embedding_document_prefix or None,
    )
