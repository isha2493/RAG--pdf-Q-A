"""Semantic search with the Chroma vector database (HNSW index).

Usage: python search.py "your question" [top_k] [pdf_filename]
"""
import sys

from embed import embed_query
from vectorstore import get_collection


def search(query: str, k: int = 5, filename: str | None = None):
    """Return the k best chunks as (score, filename, page, chunk_index, content). Higher score = better.

    If `filename` is given, only that PDF is searched.
    """
    options = {"where": {"filename": filename}} if filename else {}
    res = get_collection().query(
        query_embeddings=[embed_query(query).tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        **options,
    )
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append((1 - float(dist), meta["filename"], meta["page"], meta["chunk_index"], doc))
    return out


if __name__ == "__main__":
    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    fname = sys.argv[3] if len(sys.argv) > 3 else None
    for score, filename, page, idx, content in search(q, k, fname):
        print(f"[{score:.3f}] {filename} p{page} chunk {idx}\n{content[:300]}\n")
