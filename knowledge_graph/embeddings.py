"""
Embedding utilities for the PDE Agents Knowledge Graph.

Converts simulation run configs + results into natural-language summaries,
then embeds them via Ollama's nomic-embed-text model (768-dim vectors).

The resulting vectors are stored on Run nodes in Neo4j and used for
semantic similarity search via a HNSW vector index.

Embedding model
───────────────
nomic-embed-text is a 768-dimensional text embedding model optimised for
retrieval tasks.  At ~274 MB it loads quickly alongside the larger chat
models that are already in Ollama.

Fallback behaviour
──────────────────
If Ollama is unreachable or the model is not yet pulled, all methods
degrade gracefully: embed_text() returns None, embed_run() returns None,
and get_similar_runs_semantic() returns an empty list.  The caller always
falls back to the existing Cypher-based similarity search.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://ollama:11434")
EMBED_MODEL      = os.getenv("EMBED_MODEL",       "nomic-embed-text")
EMBED_DIMENSIONS = 768   # nomic-embed-text output dimension


# ─── Text summariser ──────────────────────────────────────────────────────────

def run_to_text(run_id: str, config: dict, results: dict) -> str:
    """
    Convert a simulation run's config + results into a descriptive text
    summary suitable for embedding.

    The summary is designed to capture the physically meaningful aspects of
    the run so that semantically similar runs (same material class, similar
    BCs, comparable domain size) score high on cosine similarity.

    Example output
    ──────────────
    "2D heat conduction in l_shape domain (Lx=0.08m, Ly=0.08m),
     k=50.0 W/(m·K) [medium_conductor], BCs: dirichlet+robin,
     component-scale geometry, T_max=800.0K T_min=753.1K T_mean=771.2K,
     DOFs=116, wall_time=0.69s, status=success."
    """
    dim  = config.get("dim",    2)    or 2
    k    = config.get("k",     1.0)  or 1.0
    rho  = config.get("rho",   1.0)  or 1.0
    cp   = config.get("cp",    1.0)  or 1.0
    Lx   = config.get("Lx",   1.0)  or 1.0
    Ly   = config.get("Ly",   1.0)  or 1.0
    Lz   = config.get("Lz",   1.0)  or 1.0
    src  = config.get("source", 0.0) or 0.0
    t_end= config.get("t_end", 1.0)  or 1.0
    dt   = config.get("dt",   0.01)  or 0.01

    # Geometry description
    geo = config.get("geometry") or {}
    if geo:
        geo_type = geo.get("type", "custom")
        geo_desc = f"{dim}D {geo_type} geometry"
        ms = geo.get("mesh_size")
        if ms:
            geo_desc += f" (mesh_size={ms}m)"
    else:
        geo_desc = f"{dim}D rectangular domain ({Lx}m × {Ly}m"
        if dim == 3:
            geo_desc += f" × {Lz}m"
        geo_desc += ")"

    # Domain size class
    import math
    char_len = math.sqrt(max(Lx, 1e-12) * max(Ly, 1e-12))
    if char_len < 0.015:
        domain_class = "micro-scale"
    elif char_len < 0.060:
        domain_class = "component-scale"
    elif char_len < 0.200:
        domain_class = "panel-scale"
    else:
        domain_class = "structural-scale"

    # Thermal class
    if k > 50:
        tc = "high_conductor"
    elif k > 10:
        tc = "medium_conductor"
    elif k > 1:
        tc = "low_conductor"
    else:
        tc = "thermal_insulator"

    # BC summary
    bcs = config.get("bcs", [])
    bc_types = sorted({b.get("type", "unknown") for b in bcs})
    bc_str = "+".join(bc_types) if bc_types else "unknown"

    # Robin parameters (convective cooling)
    robin_params = []
    for bc in bcs:
        if bc.get("type") == "robin":
            h     = bc.get("h", bc.get("alpha", "?"))
            t_inf = bc.get("T_inf", bc.get("u_inf", "?"))
            robin_params.append(f"h={h} T_inf={t_inf}K")
    robin_str = (" robin: " + ", ".join(robin_params)) if robin_params else ""

    # Source term
    src_str = f" internal heat source={src}W/m³," if src else ""

    # Results
    t_max  = results.get("max_temperature")
    t_min  = results.get("min_temperature")
    t_mean = results.get("mean_temperature")
    wt     = results.get("wall_time", 0.0)
    dofs   = results.get("n_dofs", 0)
    status = results.get("status", "unknown")

    t_str = ""
    if t_max is not None:
        t_str = f"T_max={t_max:.1f}K T_min={t_min:.1f}K T_mean={t_mean:.1f}K, "

    alpha = k / (rho * cp) if rho * cp else 0.0

    text = (
        f"{geo_desc}, {domain_class}, "
        f"k={k} W/(m·K) [{tc}], rho={rho} kg/m³, cp={cp} J/(kg·K), "
        f"thermal_diffusivity={alpha:.3e} m²/s, "
        f"BCs: {bc_str}{robin_str},{src_str} "
        f"t_end={t_end}s dt={dt}s, "
        f"{t_str}"
        f"DOFs={dofs}, wall_time={wt:.2f}s, status={status}."
    )
    return text


# ─── Embedder ─────────────────────────────────────────────────────────────────

class OllamaEmbedder:
    """
    Wraps Ollama's /api/embeddings endpoint to produce float vectors.

    Uses a module-level singleton (get_embedder()) to avoid re-instantiation.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL,
                 model: str = EMBED_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self._available: Optional[bool] = None  # None = not yet checked

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            # Accept exact name or prefix match (e.g. "nomic-embed-text:latest")
            self._available = any(
                m == self.model or m.startswith(self.model + ":")
                for m in models
            )
            if not self._available:
                log.warning(
                    "Embed model '%s' not found in Ollama. Available: %s. "
                    "Run: ollama pull %s",
                    self.model, models, self.model,
                )
        except Exception as exc:
            log.warning("Ollama not reachable for embeddings: %s", exc)
            self._available = False
        return self._available

    def embed_text(self, text: str) -> Optional[list[float]]:
        """
        Return a 768-dim embedding vector for the given text, or None on error.
        """
        if not self._check_available():
            return None
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            r.raise_for_status()
            vec = r.json().get("embedding")
            if vec and len(vec) > 0:
                return vec
            log.warning("Ollama returned empty embedding for model=%s", self.model)
            return None
        except Exception as exc:
            log.warning("embed_text failed: %s", exc)
            self._available = None  # re-check next time
            return None

    def embed_run(self, run_id: str, config: dict, results: dict) -> Optional[list[float]]:
        """Embed a simulation run summary and return the vector."""
        text = run_to_text(run_id, config, results)
        vec  = self.embed_text(text)
        if vec:
            log.debug("Embedded run %s  dims=%d", run_id, len(vec))
        return vec

    def reset_availability_cache(self) -> None:
        """Force re-check of Ollama availability on next call."""
        self._available = None


# ─── Singleton ────────────────────────────────────────────────────────────────

_embedder_instance: Optional[OllamaEmbedder] = None


def get_embedder() -> OllamaEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = OllamaEmbedder()
    return _embedder_instance
