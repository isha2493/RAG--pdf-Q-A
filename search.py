"""Semantic search over embedded chunks.

Usage: python search.py "your question" [top_k] [pdf_filename]
"""
import sys

import numpy as np

from db import get_connection
from embed import embed_query


def search(query: str, k: int = 5, filename: str | None = None):
    """Return the k best chunks as (score, filename, page, chunk_index, content).

    If `filename` is given, only that PDF is searched (so answers come from
    the PDF the user picked, not from every PDF in the database).
    """
    sql = """SELECT d.filename, c.page_number, c.chunk_index, c.content, c.embedding
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.embedding IS NOT NULL"""
    params: list = []
    if filename:
        sql += " AND d.filename = %s"
        params.append(filename)

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []
    matrix = np.array([r[4] for r in rows], dtype=np.float32)   # vectors are normalized -> dot = cosine
    scores = matrix @ embed_query(query)
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), *rows[i][:4]) for i in top]


if __name__ == "__main__":
    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    fname = sys.argv[3] if len(sys.argv) > 3 else None
    for score, filename, page, idx, content in search(q, k, fname):
        print(f"[{score:.3f}] {filename} p{page} chunk {idx}\n{content[:300]}\n")
