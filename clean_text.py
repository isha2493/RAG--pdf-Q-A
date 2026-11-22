

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HEADER_FOOTER_MIN_PAGES = 3      # need at least this many pages to detect repeats
HEADER_FOOTER_RATIO = 0.4        # line must appear on >= this share of pages
HEADER_FOOTER_MAX_LEN = 120      # headers/footers are short

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
HTML_TAG = re.compile(r"</?(?:u|b|i|em|strong|sub|sup|span)\b[^>]*>", re.IGNORECASE)
DOT_LEADER = re.compile(r"(?:\s*\.){4,}")                 # ".....", ". . . ."
TABLE_SEPARATOR = re.compile(r"^\|?(?:\s*:?-{3,}:?\s*\|?)+$")
EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1|(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\3(?![\w*])")
PAGE_NUMBER = re.compile(
    r"^(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,7})(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE
)
HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
# a "word" of mostly punctuation, 8+ long -> watermark/garbage
PUNCT_RUN = re.compile(r"[`'\"~^,.\-_=|\\/]{8,}")


def _strip_emphasis(line: str) -> str:
    return EMPHASIS.sub(lambda m: m.group(2) or m.group(4), line)


def _is_noise_line(line: str) -> bool:
    """True for lines with no alphanumeric content at all, or watermark strings."""
    s = line.strip()
    if not s:
        return False
    if not re.search(r"[A-Za-z0-9ऀ-ॿ]", s):
        return True                       
    return False


def _basic_clean_page(text: str) -> str:
    """Per-page cleanup that needs no knowledge of the other pages."""
    text = unicodedata.normalize("NFKC", text)
    text = HTML_COMMENT.sub(" ", text)
    text = HTML_BR.sub(" ", text)
    text = HTML_TAG.sub("", text)
    text = text.replace("­", "")                       # soft hyphen
    text = text.replace(" ", " ")
    text = re.sub(r"[​-‍﻿]", "", text)       # zero-width chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if TABLE_SEPARATOR.match(stripped) and "-" in stripped:
            continue
        line = PUNCT_RUN.sub(" ", line)
        line = DOT_LEADER.sub(" ", line)
        if stripped.startswith("|"):
            # flatten table row: "|a|b|c|" -> "a | b | c"
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            cells = [c for c in cells if c and not _is_noise_line(c)]
            line = " | ".join(cells)
        line = _strip_emphasis(line)
        line = line.replace("`", "") if _is_noise_line(line.replace("`", "")) else line
        if _is_noise_line(line):
            continue
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return "\n".join(lines)


def _signature(line: str) -> str:
    """Normalise a line so 'Page 3' and 'Page 4' (or '© X 2022') compare equal."""
    s = re.sub(r"^#+\s*", "", line.strip().lower())
    return re.sub(r"\d+", "#", s)


def _repeated_signatures(pages: list[str]) -> set[str]:
    if len(pages) < HEADER_FOOTER_MIN_PAGES:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        sigs = {
            _signature(l)
            for l in page.split("\n")
            if l.strip() and len(l) <= HEADER_FOOTER_MAX_LEN
        }
        counts.update(sigs)
    threshold = max(HEADER_FOOTER_MIN_PAGES, HEADER_FOOTER_RATIO * len(pages))
    return {sig for sig, n in counts.items() if n >= threshold and sig != "#"}


def _finalize(text: str) -> str:
    text = HYPHEN_BREAK.sub(r"\1", text)                    # "inter-\nnational" -> "international"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(texts: list[str]) -> list[str]:
    """Clean the parsed text of every page of ONE document.

    Takes/returns a list in page order (same length), because header/footer
    detection needs to see the whole document."""
    basic = [_basic_clean_page(t) for t in texts]
    repeated = _repeated_signatures(basic)

    cleaned = []
    for page in basic:
        kept = []
        for line in page.split("\n"):
            s = line.strip()
            if s and len(s) <= HEADER_FOOTER_MAX_LEN:
                if _signature(s) in repeated or PAGE_NUMBER.match(s):
                    continue
            kept.append(line)
        # blank lines produced by removals/noise are squashed in _finalize
        cleaned.append(_finalize("\n".join(kept)))
    return cleaned


def clean_text(text: str) -> str:
    """Single-page convenience (no header/footer detection)."""
    return _finalize(_basic_clean_page(text))


CLEAN_DIR = Path("data") / "clean"
PARSED_DIR = Path("data") / "parsed"


if __name__ == "__main__":
    # Preview: clean every data/parsed/*.raw_pages.json -> data/clean/<name>.txt
    files = [Path(a) for a in sys.argv[1:]] or sorted(PARSED_DIR.glob("*.raw_pages.json"))
    if not files:
        sys.exit(f"Nothing in {PARSED_DIR}/ - run parse_pdf.py first.")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.suffix.lower() == ".pdf":
            from parse_pdf import parse_pdf      # PDF given: parse it first
            docs = parse_pdf(str(f))
            stem = f.stem
        elif f.suffix.lower() == ".json":
            docs = json.loads(f.read_text(encoding="utf-8"))
            stem = f.name.removesuffix(".raw_pages.json").removesuffix(".json")
        else:
            sys.exit(f"Unsupported file {f} - give a .pdf or a .raw_pages.json")
        before = [d["text"] for d in docs]
        after = clean_pages(before)
        out = CLEAN_DIR / f"{stem}.txt"
        with open(out, "w", encoding="utf-8") as fh:
            for d, t in zip(docs, after):
                fh.write(f"===== PAGE {d['metadata']['page']} =====\n{t}\n\n")
        n_b, n_a = sum(map(len, before)), sum(map(len, after))
        print(f"{f.name}: {n_b} -> {n_a} chars ({100 - 100 * n_a // max(n_b, 1)}% removed) -> {out}")
