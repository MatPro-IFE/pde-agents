"""
Structured document extraction pipeline powered by IBM Docling.

Converts uploaded PDFs / DOCX / TXT into a list of contextualised chunks,
each with:
  - section heading context
  - chunk text
  - chunk type (text | table | list | equation)
  - page number (if available)
  - physics-domain classification (material, bc, solver, domain, general)

The HybridChunker produces token-aware chunks that respect document
structure (headings, tables, lists) so each chunk embeds independently
with high retrieval quality.

Fallback: If Docling cannot parse a file (e.g. scanned image-only PDF),
the pipeline falls back to pypdf flat extraction and naive chunking.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MAX_CHUNK_TOKENS = 256
OVERLAP_TOKENS   = 32


@dataclass
class DocumentChunk:
    """A single semantically meaningful piece of a parsed document."""
    chunk_index:    int
    text:           str
    heading:        str       = ""
    chunk_type:     str       = "text"      # text | table | list | equation
    page:           int       = 0
    classification: str       = "general"   # material | bc | solver | domain | general
    confidence:     float     = 0.0
    embedding:      Optional[list[float]] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class ParsedDocument:
    """Full result of structured document parsing."""
    title:      str
    n_pages:    int
    n_chunks:   int
    chunks:     list[DocumentChunk]
    tables:     list[dict]          # raw table data if extracted
    full_text:  str                 # reassembled full text for backward compat
    method:     str = "docling"     # docling | pypdf_fallback


# ── Classification keywords for physics-domain tagging ─────────────────────

_DOMAIN_PATTERNS: dict[str, list[str]] = {
    "material": [
        r"thermal\s*conductiv",  r"\bk\s*=\s*\d",  r"density", r"specific\s*heat",
        r"steel", r"copper", r"aluminum", r"titanium", r"concrete", r"glass\s*wool",
        r"polyurethane", r"ceramic", r"wood", r"alloy", r"W/\(m", r"kg/m",
        r"J/\(kg", r"diffusivity",
    ],
    "bc": [
        r"boundary\s*condition", r"dirichlet", r"neumann", r"robin",
        r"convect", r"insulated", r"heat\s*flux", r"prescribed\s*temperature",
        r"adiabatic", r"h\s*=\s*\d", r"T_?inf", r"ambient",
    ],
    "solver": [
        r"finite\s*element", r"FEM\b", r"FEA\b", r"mesh\s*refine",
        r"convergence", r"time\s*step", r"implicit", r"explicit",
        r"crank.?nicolson", r"theta.?scheme", r"CFL", r"DOF",
        r"Newton.?Raphson", r"linear\s*solver", r"precondition",
    ],
    "domain": [
        r"geometry", r"domain\s*size", r"length\s*scale", r"L-?shape",
        r"cylinder", r"rectangle", r"annul", r"spatial",
        r"2D\b", r"3D\b", r"dimension",
    ],
}

_COMPILED_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _DOMAIN_PATTERNS.items()
}


def _classify_chunk(text: str) -> tuple[str, float]:
    """Classify a chunk into a physics domain category using keyword scoring."""
    scores: dict[str, int] = {cat: 0 for cat in _COMPILED_PATTERNS}
    for cat, patterns in _COMPILED_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                scores[cat] += 1

    best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
    total_hits = scores[best_cat]
    if total_hits == 0:
        return "general", 0.0
    confidence = min(1.0, total_hits / 4.0)
    return best_cat, round(confidence, 2)


# ── Docling-based parsing ──────────────────────────────────────────────────

def _parse_with_docling(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Use Docling's DocumentConverter + HybridChunker for structured extraction.
    """
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    suffix = Path(filename).suffix.lower() or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(source=tmp_path)
        doc = result.document
    finally:
        os.unlink(tmp_path)

    chunker = HybridChunker(
        max_tokens=MAX_CHUNK_TOKENS,
        overlap=OVERLAP_TOKENS,
    )

    chunks: list[DocumentChunk] = []
    tables: list[dict] = []
    full_parts: list[str] = []

    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        enriched_text = chunker.contextualize(chunk=chunk)
        raw_text = chunk.text or ""

        heading = ""
        for meta_item in (chunk.meta.headings or []):
            heading = meta_item
            break

        chunk_type = "text"
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
            for item in (chunk.meta.doc_items or []):
                label = getattr(item, "label", "") or ""
                label_lower = label.lower()
                if "table" in label_lower:
                    chunk_type = "table"
                    tables.append({"index": i, "heading": heading, "text": raw_text})
                elif "list" in label_lower:
                    chunk_type = "list"
                elif "equation" in label_lower or "formula" in label_lower:
                    chunk_type = "equation"

        page = 0
        if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
            for item in (chunk.meta.doc_items or []):
                prov_list = getattr(item, "prov", None) or []
                for prov in prov_list:
                    page = getattr(prov, "page_no", 0) or 0
                    if page:
                        break
                if page:
                    break

        classification, conf = _classify_chunk(enriched_text)

        chunks.append(DocumentChunk(
            chunk_index=i,
            text=enriched_text,
            heading=heading,
            chunk_type=chunk_type,
            page=page,
            classification=classification,
            confidence=conf,
        ))
        full_parts.append(enriched_text)

    n_pages = 0
    if hasattr(doc, "pages"):
        n_pages = len(doc.pages) if doc.pages else 0

    title_text = filename
    if hasattr(doc, "name") and doc.name:
        title_text = doc.name

    return ParsedDocument(
        title=title_text,
        n_pages=n_pages,
        n_chunks=len(chunks),
        chunks=chunks,
        tables=tables,
        full_text="\n\n".join(full_parts),
        method="docling",
    )


# ── Fallback: pypdf + naive chunking ──────────────────────────────────────

def _parse_with_pypdf_fallback(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Simple fallback when Docling fails: pypdf + fixed-size text splitting."""
    text = ""
    n_pages = 0
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        n_pages = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        for enc in ("utf-8", "latin-1"):
            try:
                text = file_bytes.decode(enc).strip()
                break
            except UnicodeDecodeError:
                continue

    if not text:
        text = file_bytes.decode("utf-8", errors="replace").strip()

    chunk_size = 800
    overlap = 100
    chunks: list[DocumentChunk] = []
    pos = 0
    idx = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:end].strip()
        if chunk_text:
            classification, conf = _classify_chunk(chunk_text)
            chunks.append(DocumentChunk(
                chunk_index=idx,
                text=chunk_text,
                heading="",
                chunk_type="text",
                page=0,
                classification=classification,
                confidence=conf,
            ))
            idx += 1
        pos = end - overlap if end < len(text) else end

    return ParsedDocument(
        title=filename,
        n_pages=n_pages,
        n_chunks=len(chunks),
        chunks=chunks,
        tables=[],
        full_text=text,
        method="pypdf_fallback",
    )


# ── Public API ─────────────────────────────────────────────────────────────

def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    """
    Parse a document into structured, classified chunks.

    Tries Docling first (preserves tables, sections, equations).
    Falls back to pypdf + naive chunking if Docling fails.
    """
    try:
        result = _parse_with_docling(file_bytes, filename)
        if result.n_chunks > 0:
            log.info(
                "Docling parsed '%s': %d pages, %d chunks, %d tables",
                filename, result.n_pages, result.n_chunks, len(result.tables),
            )
            return result
        log.warning("Docling returned 0 chunks for '%s', falling back", filename)
    except Exception as exc:
        log.warning("Docling failed for '%s': %s — using pypdf fallback", filename, exc)

    result = _parse_with_pypdf_fallback(file_bytes, filename)
    log.info(
        "pypdf fallback parsed '%s': %d pages, %d chunks",
        filename, result.n_pages, result.n_chunks,
    )
    return result


def embed_chunks(
    chunks: list[DocumentChunk],
    batch_size: int = 8,
) -> list[DocumentChunk]:
    """
    Generate embeddings for each chunk using the Ollama embedder.
    Mutates chunks in-place and returns them.
    """
    try:
        from knowledge_graph.embeddings import get_embedder
        embedder = get_embedder()
    except Exception:
        log.warning("Embedder unavailable, skipping chunk embeddings")
        return chunks

    for chunk in chunks:
        try:
            vec = embedder.embed_text(chunk.text[:2000])
            chunk.embedding = vec
        except Exception as exc:
            log.debug("Failed to embed chunk %d: %s", chunk.chunk_index, exc)

    n_embedded = sum(1 for c in chunks if c.embedding is not None)
    log.info("Embedded %d / %d chunks", n_embedded, len(chunks))
    return chunks
