"""Look inside your databases.

  python view_db.py                  -> summary: what PDFs are stored and how many pieces each has
  python view_db.py file.pdf         -> show the first 10 pieces (chunks) of that PDF, with their vectors
  python view_db.py file.pdf 30      -> show the first 30 pieces
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # so Nepali text never crashes the terminal

from db import get_connection
from vectorstore import HNSW, count_for, get_collection

args = sys.argv[1:]

if not args:
    col = get_collection()
    print("=== CHROMA (vector database) ===")
    print(f"total vectors : {col.count()}")
    print(f"HNSW settings : {HNSW}")
    metas = col.get(include=["metadatas"])["metadatas"]
    per_pdf: dict[str, int] = {}
    for m in metas:
        per_pdf[m["filename"]] = per_pdf.get(m["filename"], 0) + 1
    for name, n in sorted(per_pdf.items()):
        print(f"  {name}: {n} chunks")

    print("\n=== POSTGRES (text) ===")
    for table in ("documents", "pages", "chunks"):
        try:
            with get_connection() as conn:
                n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {n} rows")
        except Exception as e:
            print(f"  {table}: could not read ({str(e).splitlines()[0]})")
    print("\nTip: python view_db.py yourfile.pdf   to see the chunks of one PDF")
else:
    filename = args[0]
    limit = int(args[1]) if len(args) > 1 else 10
    total = count_for(filename)
    if total == 0:
        print(f"No vectors found for '{filename}'. Run  python view_db.py  to see the exact names.")
        sys.exit(1)
    data = get_collection().get(
        where={"filename": filename}, limit=limit, include=["documents", "metadatas", "embeddings"]
    )
    print(f"{filename}: {total} chunks in total, showing {len(data['ids'])}\n")
    for cid, doc, meta, vec in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]):
        first = ", ".join(f"{x:.3f}" for x in vec[:5])
        print(f"--- chunk id {cid} | page {meta['page']} | chunk {meta['chunk_index']}")
        print(f"text   : {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print(f"vector : [{first}, ...]  ({len(vec)} numbers)\n")