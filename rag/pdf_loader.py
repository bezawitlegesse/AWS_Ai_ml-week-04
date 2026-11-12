"""Read a PDF page by page, clean the text and cut it into overlapping chunks (page number kept)."""
import re
from dataclasses import dataclass


@dataclass
class Page:
    number: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str


def clean_text(text):
    """Fix common PDF text problems: hyphenated line breaks, broken lines, extra spaces."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)            # re-join words split across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single line breaks become spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(path):
    """Return a list of Page objects (pages without text are skipped)."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(Page(number, text))
    if not pages:
        raise ValueError(
            "No text could be extracted. The PDF may be scanned images; this project needs a text-based PDF.")
    return pages


def chunk_text(text, size=500, overlap=50):
    """Cut text into chunks of about `size` characters that share `overlap` characters."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                            # prefer to cut at a sentence or word boundary
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(pages, size=500, overlap=50):
    """Chunk every page and remember which page each chunk came from."""
    chunks = []
    for page in pages:
        for index, piece in enumerate(chunk_text(page.text, size, overlap)):
            chunks.append(Chunk(f"p{page.number}-c{index}", page.number, piece))
    return chunks
