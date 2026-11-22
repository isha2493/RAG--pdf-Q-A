
from pathlib import Path
import json

import pymupdf
import pymupdf4llm

MIN_CHARS_PER_PAGE = 50  



def parse_pdf(path: str) -> list[dict]:
    pages = pymupdf4llm.to_markdown(path, page_chunks=True)
    with pymupdf.open(path) as pdf:
        raw_pages = [page.get_text() for page in pdf]
    docs = []
    for i, p in enumerate(pages, start=1):
        text = p["text"]
        if len(text) < MIN_CHARS_PER_PAGE:
            print(f"  warning: page {i} has little text (scanned? needs OCR)")
        docs.append(
            {
                "raw_text": raw_pages[i - 1],
                "text": text,
                "metadata": {"source": Path(path).name, "page": i},
            }
        )
    return docs



DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "parsed"


def find_pdfs(args: list[str]) -> list[Path]:
    """PDFs named on the command line, or every PDF in data/ if none given."""
    if args:
        return [Path(a) for a in args]
    return sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf")


if __name__ == "__main__":
    import sys

    pdfs = find_pdfs(sys.argv[1:])
    if not pdfs:
        sys.exit(f"No PDFs found in {DATA_DIR}/ - drop one there and rerun.")

    OUT_DIR.mkdir(exist_ok=True)
    for pdf in pdfs:
        print(f"Parsing {pdf.name} ...")
        docs = parse_pdf(str(pdf))

        json_path = OUT_DIR / f"{pdf.stem}.raw_pages.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        print(f"  {len(docs)} pages -> {json_path}")

