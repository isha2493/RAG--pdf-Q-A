"""Turn every chunk into a vector (list of 384 numbers) and store it in the Chroma vector database.

Postgres keeps the chunk text. This script keeps Chroma in step with Postgres:
  - chunks that have no vector yet  -> embedded and added
  - vectors whose chunk no longer exists in Postgres (PDF was reloaded) -> deleted
Re-running is always safe.
"""
from db import get_connection
from vectorstore import embedded_ids, get_collection

MODEL_NAME = "intfloat/multilingual-e5-small"   # 384 numbers per chunk, English + Nepali
BATCH = 32

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]):
    # e5 models expect a "passage: " prefix on documents
    return get_model().encode(
        [f"passage: {t}" for t in texts], batch_size=BATCH, normalize_embeddings=True
    )


def embed_query(text: str):
    return get_model().encode(f"query: {text}", normalize_embeddings=True)


def main() -> None:
    col = get_collection()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT c.id, d.filename, c.page_number, c.chunk_index, c.content
               FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.id"""
        ).fetchall()

    in_postgres = {str(r[0]) for r in rows}
    in_chroma = embedded_ids()

    stale = list(in_chroma - in_postgres)                 # vectors of chunks that were deleted
    for i in range(0, len(stale), 500):
        col.delete(ids=stale[i:i + 500])
    if stale:
        print(f"removed {len(stale)} old vectors")

    todo = [r for r in rows if str(r[0]) not in in_chroma]
    print(f"{len(todo)} chunks to embed with {MODEL_NAME}")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vecs = embed_passages([r[4] for r in batch])
        col.add(
            ids=[str(r[0]) for r in batch],
            embeddings=vecs.tolist(),
            documents=[r[4] for r in batch],
            metadatas=[{"filename": r[1], "page": r[2], "chunk_index": r[3]} for r in batch],
        )
        print(f"  embedded {min(i + BATCH, len(todo))}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    main()
