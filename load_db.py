
import sys

from db import get_connection
from chunk import chunk_text
from clean_text import clean_pages
from parse_pdf import parse_pdf

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    filename    TEXT UNIQUE NOT NULL
);

ALTER TABLE documents DROP COLUMN IF EXISTS uploaded_at;
ALTER TABLE documents DROP COLUMN IF EXISTS num_pages;

CREATE TABLE IF NOT EXISTS pages (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL REFERENCES documents(filename) ON UPDATE CASCADE ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    raw_text    TEXT NOT NULL,
    parsed_text TEXT NOT NULL,
    clean_text  TEXT,
    UNIQUE (document_id, page_number)
);

-- upgrades for databases created with the earlier schema (rows are refilled on the next load)
ALTER TABLE pages ADD COLUMN IF NOT EXISTS filename TEXT
    REFERENCES documents(filename) ON UPDATE CASCADE ON DELETE CASCADE;
ALTER TABLE pages ADD COLUMN IF NOT EXISTS clean_text TEXT;

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    UNIQUE (document_id, page_number, chunk_index)
);
"""

UPSERT_DOCUMENT = """
INSERT INTO documents (filename)
VALUES (%s)
ON CONFLICT (filename) DO UPDATE SET filename = EXCLUDED.filename
RETURNING id;
"""

UPSERT_PAGE = """
INSERT INTO pages (document_id, filename, page_number, raw_text, parsed_text, clean_text)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (document_id, page_number)
DO UPDATE SET filename = EXCLUDED.filename, raw_text = EXCLUDED.raw_text, parsed_text = EXCLUDED.parsed_text,
    clean_text = EXCLUDED.clean_text;
"""

INSERT_CHUNK = """
INSERT INTO chunks (document_id, page_number, chunk_index, content)
VALUES (%s, %s, %s, %s);
"""


def load_pdf(path: str) -> None:
    pages = parse_pdf(path)                                 
    filename = pages[0]["metadata"]["source"]
    clean = clean_pages([p["text"] for p in pages])

    with get_connection() as conn:                         
        conn.execute(SCHEMA)
        document_id = conn.execute(UPSERT_DOCUMENT, (filename,)).fetchone()[0]
        for p, clean_page in zip(pages, clean):
            conn.execute(
                UPSERT_PAGE,
                (document_id, filename, p["metadata"]["page"], p["raw_text"], p["text"], clean_page),
            )

        # re-chunk from scratch so reloading a PDF never leaves stale chunks
        conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        n_chunks = 0
        for p, clean_page in zip(pages, clean):
            for i, content in enumerate(chunk_text(clean_page)):
                conn.execute(INSERT_CHUNK, (document_id, p["metadata"]["page"], i, content))
                n_chunks += 1

    print(f"Loaded {filename} (document id {document_id}): {len(pages)} pages, {n_chunks} chunks")


if __name__ == "__main__":
    load_pdf(sys.argv[1] if len(sys.argv) > 1 else "data/rag-ios.pdf")
