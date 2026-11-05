"""Read files from data/ (.pdf, .md, .txt), chunk them, embed them, store them in ChromaDB."""
import bisect
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import store
from config import DATA_DIR

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def load_file(path: Path):
    """Return (full_text, page_starts). page_starts maps text offsets to page numbers."""
    if path.suffix.lower() == ".pdf":
        parts, starts, pos = [], [], 0
        for page_no, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            starts.append((pos, page_no))
            parts.append(text)
            pos += len(text) + 2
        return "\n\n".join(parts), starts
    return path.read_text(encoding="utf-8"), [(0, 1)]


def build_chunks(path: Path):
    text, starts = load_file(path)
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    offsets = [s[0] for s in starts]
    chunks = []
    for i, doc in enumerate(splitter.create_documents([text])):
        start = max(doc.metadata.get("start_index", 0), 0)
        page = starts[max(bisect.bisect_right(offsets, start) - 1, 0)][1]
        chunks.append({"id": f"{path.name}-c{i}", "text": doc.page_content, "source": path.name, "page": page})
    return chunks


def main():
    files = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in {".pdf", ".md", ".txt"})
    if not files:
        sys.exit(f"No .pdf/.md/.txt files found in {DATA_DIR}")
    chunks = []
    for f in files:
        part = build_chunks(f)
        print(f"{f.name}: {len(part)} chunks")
        chunks += part
    col = store.reset_collection()
    store.add_chunks(col, chunks)
    print(f"Done: {len(files)} files, {len(chunks)} chunks stored.")


if __name__ == "__main__":
    main()
