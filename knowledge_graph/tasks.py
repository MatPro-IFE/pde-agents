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
