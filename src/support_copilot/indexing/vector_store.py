"""Own the persistent Chroma client and collection reset behavior."""

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


# COLLECTION_NAME = "technical_evidence"

# Explicit, not cosmetic: Chroma defaults to "l2" (squared Euclidean) when unset. That only ranks
# the same as cosine similarity because the current embedding model (nomic-embed-text) happens to
# emit unit-normalized vectors — an assumption this constant now makes explicit instead of leaving
# incidental. See docs/learning/vector-similarity-scoring.md. `hnsw:space` is fixed at collection
# creation time; it is part of index_manager.vector_policy(), so changing it triggers a full
# rebuild on the next ingestion run.
DISTANCE_SPACE = "cosine"


def get_collection(config, *, reset: bool = False):
    """Return the Chroma collection, optionally replacing all generated vectors."""

    config.storage_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.storage_dir / "chroma_db"))
    if reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            # Chroma raises when the collection does not yet exist; first-run ingestion is healthy.
            pass
    return client.get_or_create_collection(config.collection_name, metadata={"hnsw:space": DISTANCE_SPACE})


def get_vector_store(config, *, reset: bool = False):
    """Wrap Chroma in the adapter expected by LlamaIndex."""

    return ChromaVectorStore(chroma_collection=get_collection(config, reset=reset))
