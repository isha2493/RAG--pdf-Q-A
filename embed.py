
from db import get_connection

MODEL_NAME = "intfloat/multilingual-e5-small"   
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
    with get_connection() as conn:
        conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding REAL[]")
        rows = conn.execute("SELECT id, content FROM chunks WHERE embedding IS NULL ORDER BY id").fetchall()
        print(f"{len(rows)} chunks to embed with {MODEL_NAME}")
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            vecs = embed_passages([r[1] for r in batch])
            for (chunk_id, _), v in zip(batch, vecs):
                conn.execute("UPDATE chunks SET embedding = %s WHERE id = %s", (v.tolist(), chunk_id))
            print(f"  embedded {min(i + BATCH, len(rows))}/{len(rows)}")
    print("done")


if __name__ == "__main__":
    main()
