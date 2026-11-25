"""The vector database (ChromaDB, HNSW search). Everything else talks to Chroma through this file.

Chroma saves its data in a folder called chroma_db/ next to this file. No server, no install steps.
Postgres still keeps the text of your PDFs; Chroma keeps the vectors (the "meaning numbers").
"""
from pathlib import Path

import chromadb

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION = "chunks"

# HNSW settings (think of HNSW as a map with highways on top and small streets below)
HNSW = {
    "space": "cosine",        # compare by meaning-angle (our vectors are built for this)
    "max_neighbors": 16,      # links per point: more = more accurate, bigger index
    "ef_construction": 100,   # how carefully the index is built: more = better quality, slower build
    "ef_search": 100,         # how many candidates are checked per search: more = more accurate, slower
}

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        _collection = client.get_or_create_collection(
            COLLECTION, embedding_function=None, configuration={"hnsw": HNSW}
        )
    return _collection


def embedded_ids() -> set[str]:
    """ids of every chunk that already has a vector."""
    return set(get_collection().get(include=[])["ids"])


def embedded_filenames() -> set[str]:
    """Names of the PDFs that have vectors."""
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    return {m["filename"] for m in metas}


def count_for(filename: str) -> int:
    """How many vectors this PDF has."""
    return len(get_collection().get(where={"filename": filename}, include=[])["ids"])
