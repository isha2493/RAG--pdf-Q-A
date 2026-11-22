MAX_CHARS = 800   # target chunk size
OVERLAP = 100     # characters repeated from the end of the previous chunk


def chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split text into chunks by packing whole paragraphs up to max_chars."""
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # a paragraph longer than max_chars is cut into pieces first
        step = max_chars - overlap
        pieces = [para[i:i + max_chars] for i in range(0, len(para), step)] if len(para) > max_chars else [para]
        for piece in pieces:
            if current and len(current) + len(piece) > max_chars:
                chunks.append(current)
                tail = current[-overlap:]
                tail = tail[tail.find(" ") + 1:]                # don't start mid-word
                current = tail + "\n\n" + piece
            else:
                current = current + "\n\n" + piece if current else piece
    if current:
        chunks.append(current)
    return chunks


if __name__ == "__main__":
    import re, sys

    # input: a cleaned file from data/clean/<name>.txt (pages marked "===== PAGE n =====")
    raw = open(sys.argv[1], encoding="utf-8", newline=None).read()   # newline=None -> CRLF becomes \n
    parts = re.split(r"^===== PAGE (\d+) =====\n", raw, flags=re.MULTILINE)
    pages = list(zip(parts[1::2], parts[2::2]))                       # [(page_no, text), ...]
    total = 0

    with open("chunks.txt", "w", encoding="utf-8") as f:
        for page, text in pages:
            chunks = chunk_text(text)
            total += len(chunks)

            for i, c in enumerate(chunks):
                f.write(
                    f"--- page {page} chunk {i} ({len(c)} chars) ---\n{c}\n\n"
                )

    print(f"{total} chunks saved to chunks.txt")
