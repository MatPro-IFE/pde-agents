"""
Celery tasks for asynchronous document ingestion.

Heavy operations — Docling PDF parsing, embedding generation, and
chunk-level KG cross-referencing — run in a Celery worker process
so the FastAPI upload endpoint can return immediately.

Task flow:
  1. API receives file → saves to MinIO → enqueues ingest_document_task
  2. Worker picks up task:
     a. Parse document with Docling (tables, sections, equations)
     b. Classify each chunk (material / bc / solver / domain / general)
     c. Embed each chunk via Ollama nomic-embed-text
     d. Store ReferenceChunk nodes in Neo4j with embeddings
     e. Cross-reference chunks to similar Runs via vector search
     f. Link chunks to matching Material/BCConfig/Domain entities
  3. Task updates Reference node with processing status
"""

from __future__ import annotations

import logging
import os

from celery import Celery

log = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "pde_agents",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,
    task_time_limit=900,
)


@celery_app.task(bind=True, name="ingest_document", max_retries=2)
def ingest_document_task(
    self,
    ref_id: str,
    file_bytes_hex: str,
    filename: str,
    auto_link_top_k: int = 5,
    min_score: float = 0.78,
) -> dict:
    """
    Parse, chunk, embed, and cross-reference an uploaded document.

    file_bytes_hex is the hex-encoded file content (JSON-safe transport).
    """
    try:
        file_bytes = bytes.fromhex(file_bytes_hex)
        log.info("Starting structured ingestion for ref=%s (%s, %d bytes)",
                 ref_id, filename, len(file_bytes))

        # ── 1. Structured parsing ──────────────────────────────────────────
        from knowledge_graph.document_processor import parse_document, embed_chunks
        parsed = parse_document(file_bytes, filename)

        # Update Reference node with parse metadata
        from knowledge_graph.graph import get_kg
        kg = get_kg()
        if kg.available:
            kg._run(
                """
                MATCH (ref:Reference {ref_id: $ref_id})
                SET ref.n_pages        = $n_pages,
                    ref.n_chunks       = $n_chunks,
                    ref.n_tables       = $n_tables,
                    ref.parse_method   = $method,
                    ref.process_status = 'embedding'
                """,
                ref_id=ref_id, n_pages=parsed.n_pages,
                n_chunks=parsed.n_chunks, n_tables=len(parsed.tables),
                method=parsed.method,
            )

        # ── 2. Embed each chunk ────────────────────────────────────────────
        embed_chunks(parsed.chunks)

        # ── 3. Store chunks and cross-reference in KG ──────────────────────
        result = {"chunks_stored": 0, "chunks_embedded": 0, "cross_refs_created": 0}
        if kg.available:
            result = kg.ingest_document_chunks(
                ref_id=ref_id,
                chunks=parsed.chunks,
                auto_link_top_k=auto_link_top_k,
                min_score=min_score,
            )

            kg._run(
                """
                MATCH (ref:Reference {ref_id: $ref_id})
                SET ref.process_status  = 'completed',
                    ref.chunks_stored   = $cs,
                    ref.chunks_embedded = $ce,
                    ref.cross_refs      = $cr
                """,
                ref_id=ref_id, cs=result.get("chunks_stored", 0),
                ce=result.get("chunks_embedded", 0),
                cr=result.get("cross_refs_created", 0),
            )

        log.info(
            "Document ingestion complete for ref=%s: %d chunks, %d embedded, %d cross-refs",
            ref_id, result.get("chunks_stored", 0),
            result.get("chunks_embedded", 0),
            result.get("cross_refs_created", 0),
        )
        return {
            "ref_id": ref_id,
            "parse_method": parsed.method,
            "n_pages": parsed.n_pages,
            **result,
        }

    except Exception as exc:
        log.exception("Document ingestion failed for ref=%s: %s", ref_id, exc)
        try:
            from knowledge_graph.graph import get_kg
            kg = get_kg()
            if kg.available:
                kg._run(
                    """
                    MATCH (ref:Reference {ref_id: $ref_id})
                    SET ref.process_status = 'failed',
                        ref.process_error  = $err
                    """,
                    ref_id=ref_id, err=str(exc)[:500],
                )
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=30)


@celery_app.task(bind=True, name="ingest_web_resource", max_retries=1,
                 soft_time_limit=1800, time_limit=2100)
def ingest_web_resource_task(
    self,
    ref_id: str,
    root_url: str,
    title: str,
    max_pages: int = 50,
    auto_link_top_k: int = 5,
    min_score: float = 0.78,
) -> dict:
    """
    Crawl a web resource (tutorial, ebook, docs), extract structured chunks
    from each page with Docling, embed them, and cross-reference to simulation runs.
    """
    from datetime import datetime, timezone

    try:
        from knowledge_graph.graph import get_kg
        from knowledge_graph.web_fetcher import fetch_and_parse_site
        from knowledge_graph.document_processor import embed_chunks

        kg = get_kg()

        log.info("Starting web ingestion: ref=%s url=%s max_pages=%d",
                 ref_id, root_url, max_pages)

        if kg.available:
            kg._run(
                """
                MATCH (ref:Reference {ref_id: $ref_id})
                SET ref.process_status = 'crawling'
                """,
                ref_id=ref_id,
            )

        site_data = fetch_and_parse_site(root_url, max_pages=max_pages)

        total_stored = 0
        total_embedded = 0
        total_xrefs = 0
        page_summaries = []

        if kg.available:
            kg._run(
                """
                MATCH (ref:Reference {ref_id: $ref_id})
                SET ref.process_status = 'embedding',
                    ref.n_pages        = $n_pages,
                    ref.total_chunks   = $n_chunks
                """,
                ref_id=ref_id, n_pages=site_data["total_pages"],
                n_chunks=site_data["total_chunks"],
            )

        for page in site_data["pages"]:
            page_url    = page["url"]
            page_title  = page["title"]
            page_chunks = page["chunks"]

            embed_chunks(page_chunks)

            # Use a sub-ref_id per page so chunks are traceable
            page_slug = (page_url.split("//", 1)[-1]
                         .replace("/", "_").replace(".", "_")[:60].strip("_"))
            page_ref_id = f"{ref_id}__page_{page_slug}"

            # Create a child Reference node for this page
            now = datetime.now(timezone.utc).isoformat()
            if kg.available:
                kg._run(
                    """
                    MATCH (parent:Reference {ref_id: $parent_id})
                    MERGE (page:Reference {ref_id: $page_ref_id})
                    SET page.title       = $title,
                        page.url         = $url,
                        page.type        = 'web_page',
                        page.is_uploaded = true,
                        page.parent_ref  = $parent_id,
                        page.created_at  = $now
                    MERGE (parent)-[:HAS_PAGE]->(page)
                    """,
                    parent_id=ref_id, page_ref_id=page_ref_id,
                    title=page_title, url=page_url, now=now,
                )

                result = kg.ingest_document_chunks(
                    ref_id=page_ref_id,
                    chunks=page_chunks,
                    auto_link_top_k=auto_link_top_k,
                    min_score=min_score,
                )
                total_stored   += result.get("chunks_stored", 0)
                total_embedded += result.get("chunks_embedded", 0)
                total_xrefs    += result.get("cross_refs_created", 0)

            page_summaries.append({
                "url": page_url,
                "title": page_title,
                "chunks": len(page_chunks),
            })

        if kg.available:
            kg._run(
                """
                MATCH (ref:Reference {ref_id: $ref_id})
                SET ref.process_status  = 'completed',
                    ref.chunks_stored   = $cs,
                    ref.chunks_embedded = $ce,
                    ref.cross_refs      = $cr,
                    ref.n_chunks        = $cs
                """,
                ref_id=ref_id, cs=total_stored,
                ce=total_embedded, cr=total_xrefs,
            )

        log.info(
            "Web ingestion complete: ref=%s pages=%d chunks=%d embedded=%d xrefs=%d",
            ref_id, len(page_summaries), total_stored, total_embedded, total_xrefs,
        )

        return {
            "ref_id": ref_id,
            "root_url": root_url,
            "pages_crawled": len(page_summaries),
            "chunks_stored": total_stored,
            "chunks_embedded": total_embedded,
            "cross_refs_created": total_xrefs,
            "pages": page_summaries,
        }

    except Exception as exc:
        log.exception("Web ingestion failed for ref=%s: %s", ref_id, exc)
        try:
            from knowledge_graph.graph import get_kg
            kg = get_kg()
            if kg.available:
                kg._run(
                    """
                    MATCH (ref:Reference {ref_id: $ref_id})
                    SET ref.process_status = 'failed',
                        ref.process_error  = $err
                    """,
                    ref_id=ref_id, err=str(exc)[:500],
                )
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=60)
