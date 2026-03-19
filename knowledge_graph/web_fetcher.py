"""
Web resource fetcher for indexing online ebooks, tutorials, and documentation.

Crawls a root URL, discovers linked pages within the same domain path,
fetches each page, and pipes the HTML through the Docling structured
extraction pipeline to produce embedded, classified, cross-referenced
ReferenceChunk nodes in the knowledge graph.

Typical targets:
  - FEniCSx tutorial  (jsdokken.com/dolfinx-tutorial/)
  - FEniCS Book chapters
  - ASHRAE digital handbooks
  - University lecture notes on FEM / heat transfer
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "pde-agents/2.0 (knowledge-graph-indexer)",
    "Accept": "text/html,application/xhtml+xml,*/*",
}

MAX_PAGES = 80
FETCH_DELAY = 1.0  # seconds between requests (polite crawling)


@dataclass
class FetchedPage:
    url: str
    title: str
    html: str
    content_length: int


def _normalise_url(url: str) -> str:
    """Strip fragment for deduplication; preserve trailing slash on directory URLs."""
    parsed = urlparse(url)
    path = parsed.path
    # Only strip trailing slash from file-like paths (not directories)
    if path.endswith("/") and len(path) > 1:
        pass  # keep the slash — it's a directory
    elif not path:
        path = "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def discover_pages(
    root_url: str,
    max_pages: int = MAX_PAGES,
    same_path_prefix: bool = True,
) -> list[str]:
    """
    BFS crawl from root_url, discovering linked pages within the same domain.

    If same_path_prefix is True, only follows links whose path starts with
    the root URL's path (e.g. /dolfinx-tutorial/*).
    """
    parsed_root = urlparse(root_url)
    root_domain = parsed_root.netloc
    # For path prefix matching, use the directory portion
    rp = parsed_root.path
    if rp.endswith(".html") or rp.endswith(".htm"):
        root_path = rp.rsplit("/", 1)[0]
    else:
        root_path = rp.rstrip("/")

    visited: set[str] = set()
    queue: list[str] = [_normalise_url(root_url)]
    ordered: list[str] = []

    while queue and len(ordered) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            ct = resp.headers.get("content-type", "")
            if "text/html" not in ct and "xhtml" not in ct:
                continue
        except Exception as exc:
            log.debug("Skip %s: %s", url, exc)
            continue

        ordered.append(url)

        soup = BeautifulSoup(resp.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith(("#", "mailto:", "javascript:")):
                continue
            absolute = _normalise_url(urljoin(url, href))
            parsed = urlparse(absolute)
            if parsed.netloc != root_domain:
                continue
            if same_path_prefix and not parsed.path.startswith(root_path):
                continue
            if absolute not in visited:
                queue.append(absolute)

        time.sleep(FETCH_DELAY)

    log.info("Discovered %d pages from %s", len(ordered), root_url)
    return ordered


def fetch_page(url: str) -> Optional[FetchedPage]:
    """Fetch a single URL and return its HTML + metadata."""
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        return FetchedPage(
            url=url, title=title or url,
            html=html, content_length=len(html),
        )
    except Exception as exc:
        log.warning("Failed to fetch %s: %s", url, exc)
        return None


def parse_html_with_docling(html: str, url: str) -> Optional[object]:
    """
    Parse HTML content through Docling's DocumentConverter + HybridChunker.

    Returns a ParsedDocument (from document_processor) or None on failure.
    """
    try:
        from knowledge_graph.document_processor import (
            _parse_with_docling,
            _parse_with_pypdf_fallback,
            _classify_chunk,
            DocumentChunk,
            ParsedDocument,
        )
        from docling.document_converter import DocumentConverter
        from docling.chunking import HybridChunker

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                          encoding="utf-8") as tmp:
            tmp.write(html)
            tmp_path = tmp.name

        try:
            converter = DocumentConverter()
            result = converter.convert(source=tmp_path)
            doc = result.document
        finally:
            os.unlink(tmp_path)

        chunker = HybridChunker(max_tokens=256, overlap=32)
        chunks: list[DocumentChunk] = []
        tables: list[dict] = []
        full_parts: list[str] = []

        for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            enriched = chunker.contextualize(chunk=chunk)
            raw_text = chunk.text or ""

            heading = ""
            for meta_item in (chunk.meta.headings or []):
                heading = meta_item
                break

            chunk_type = "text"
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                for item in (chunk.meta.doc_items or []):
                    label = getattr(item, "label", "") or ""
                    ll = label.lower()
                    if "table" in ll:
                        chunk_type = "table"
                        tables.append({"index": i, "heading": heading, "text": raw_text})
                    elif "list" in ll:
                        chunk_type = "list"
                    elif "code" in ll or "program" in ll:
                        chunk_type = "code"
                    elif "equation" in ll or "formula" in ll:
                        chunk_type = "equation"

            classification, conf = _classify_chunk(enriched)

            chunks.append(DocumentChunk(
                chunk_index=i, text=enriched, heading=heading,
                chunk_type=chunk_type, page=0,
                classification=classification, confidence=conf,
            ))
            full_parts.append(enriched)

        n_pages = 1
        title_text = url
        if hasattr(doc, "name") and doc.name:
            title_text = doc.name

        return ParsedDocument(
            title=title_text, n_pages=n_pages,
            n_chunks=len(chunks), chunks=chunks,
            tables=tables, full_text="\n\n".join(full_parts),
            method="docling_html",
        )

    except Exception as exc:
        log.warning("Docling HTML parse failed for %s: %s", url, exc)
        return None


def fetch_and_parse_site(
    root_url: str,
    max_pages: int = MAX_PAGES,
) -> dict:
    """
    High-level: crawl a site, parse each page, return all chunks grouped by page.

    Returns dict with:
      pages: list of {url, title, n_chunks, chunks: [DocumentChunk]}
      total_pages, total_chunks
    """
    urls = discover_pages(root_url, max_pages=max_pages)
    pages = []
    total_chunks = 0

    for url in urls:
        page = fetch_page(url)
        if not page:
            continue

        parsed = parse_html_with_docling(page.html, page.url)
        if not parsed or parsed.n_chunks == 0:
            continue

        pages.append({
            "url": page.url,
            "title": page.title,
            "n_chunks": parsed.n_chunks,
            "chunks": parsed.chunks,
            "tables": parsed.tables,
        })
        total_chunks += parsed.n_chunks
        log.info("Parsed %s: %d chunks", page.url, parsed.n_chunks)

    return {
        "root_url": root_url,
        "total_pages": len(pages),
        "total_chunks": total_chunks,
        "pages": pages,
    }
