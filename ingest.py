"""One-call pipeline for a newly uploaded PDF:
save file -> parse -> clean -> chunk -> store in Postgres -> embed.

Used by app.py. You can also run it by hand:  python ingest.py data/rag-ios.pdf
"""
import sys
from pathlib import Path

from db import get_connection
from embed import main as embed_all_new_chunks
from load_db import load_pdf

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


def ingest_pdf_file(path: str) -> str:
    """Run the full pipeline on a PDF that is already on disk. Returns its filename."""
    load_pdf(path)                # parse + clean + chunk + save to the database
    embed_all_new_chunks()        # embeds only chunks that have no embedding yet
    name = Path(path).name

    with get_connection() as conn:
        n = conn.execute(
            """SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.filename = %s AND c.embedding IS NOT NULL""",
            (name,),
        ).fetchone()[0]
    if n == 0:
        raise ValueError(
            "No readable text found in this PDF. It may be a scanned PDF (images only) that needs OCR."
        )
    return name


def ingest_pdf_bytes(data: bytes, original_name: str) -> str:
    """Save uploaded bytes into data/uploads/ and run the pipeline on them."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / Path(original_name).name      # .name strips any folder parts
    path.write_bytes(data)
    return ingest_pdf_file(str(path))


def list_documents() -> list[str]:
    """Filenames of PDFs that are fully ready to be asked questions (newest first)."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT d.filename FROM documents d
                   WHERE EXISTS (SELECT 1 FROM chunks c
                                 WHERE c.document_id = d.id AND c.embedding IS NOT NULL)
                   ORDER BY d.id DESC"""
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []     # tables don't exist yet (nothing uploaded so far)


if __name__ == "__main__":
    print("Ready:", ingest_pdf_file(sys.argv[1] if len(sys.argv) > 1 else "data/rag-ios.pdf"))
