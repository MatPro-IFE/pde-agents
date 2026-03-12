"""
SimulationKnowledgeGraph — Phase 2 implementation with vector embeddings.

Graph schema
────────────
Nodes
  (:Run)          — one per simulation run  (embedding: 768-dim vector)
  (:Material)     — engineering materials with thermal properties
  (:KnownIssue)   — documented failure patterns
  (:BCConfig)     — boundary-condition pattern (dirichlet, neumann, robin combos)
  (:Domain)       — physical domain size class (micro / component / panel / structural)
  (:ThermalClass) — material conductivity class (high / medium / low / insulator)

Relationships
  (:Run)-[:USES_MATERIAL {confidence}]->(:Material)
  (:Run)-[:TRIGGERED {detected_at}]->(:KnownIssue)
  (:Run)-[:USES_BC_CONFIG]->(:BCConfig)
  (:Run)-[:ON_DOMAIN]->(:Domain)
  (:Material)-[:HAS_THERMAL_CLASS]->(:ThermalClass)
  (:Run)-[:SPAWNED_FROM]->(:Run)           — created from a suggestion
  (:Run)-[:IMPROVED_OVER {metric}]->(:Run) — better result than parent
  (:Run)-[:SIMILAR_TO {score}]->(:Run)     — KNN semantic similarity (Feature 2)

Vector search
─────────────
Each Run node stores an `embedding` property (768-dim float list) produced
by nomic-embed-text via Ollama.  A Neo4j HNSW vector index on Run.embedding
enables sub-millisecond semantic similarity search.

get_similar_runs() uses vector search when Ollama + the index are available,
and falls back transparently to the Cypher parameter-distance query.

All operations degrade gracefully if Neo4j is unavailable — agents still
work, they just don't get knowledge graph context.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

log = logging.getLogger(__name__)

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "pde_graph_secret")

# ─── BC-pattern descriptions ──────────────────────────────────────────────────
_BC_DESCRIPTIONS: dict[str, str] = {
    "dirichlet":               "Fixed temperature on all boundaries",
    "neumann":                 "Pure heat-flux / insulation on all boundaries",
    "robin":                   "Fully convective (Robin) boundaries",
    "dirichlet+neumann":       "Fixed temperature with insulated or heat-flux edges",
    "dirichlet+robin":         "Fixed temperature edges with convective cooling",
    "neumann+robin":           "Heat-flux sources with convective boundaries",
    "dirichlet+neumann+robin": "Mixed: fixed temp, heat-flux, and convective boundaries",
}

# ─── Domain size classification ───────────────────────────────────────────────
# Characteristic length = sqrt(Lx * Ly).  Thresholds in metres.
_DOMAIN_THRESHOLDS = [
    (0.015, "micro",      "Micro-scale domain  (<  1.5 cm characteristic length)"),
    (0.060, "component",  "Component-scale     (1.5–6 cm characteristic length)"),
    (0.200, "panel",      "Panel-scale         (6–20 cm characteristic length)"),
    (float("inf"), "structural", "Structural-scale (> 20 cm characteristic length)"),
]

# ─── Thermal conductivity class ───────────────────────────────────────────────
# Classified by k (W/m·K) — more intuitive than α for engineering use.
_THERMAL_THRESHOLDS = [
    (50,   "high_conductor",   "k > 50 W/m·K — metals like copper, aluminium, silicon"),
    (10,   "medium_conductor", "10 < k ≤ 50 W/m·K — engineering steels, titanium alloys"),
    (1,    "low_conductor",    "1 < k ≤ 10 W/m·K — concrete, glass, ceramics"),
    (0,    "thermal_insulator","k ≤ 1 W/m·K — water, air, polymer foams"),
]


def _domain_label(Lx: float, Ly: float) -> tuple[str, str]:
    """Return (label, description) for a domain of size Lx × Ly metres."""
    char_len = math.sqrt(max(Lx, 1e-12) * max(Ly, 1e-12))
    for threshold, label, desc in _DOMAIN_THRESHOLDS:
        if char_len < threshold:
            return label, desc
    return "structural", _DOMAIN_THRESHOLDS[-1][2]


def _thermal_class(k: float | None) -> tuple[str, str] | None:
    """Return (class_name, description) for thermal conductivity k."""
    if k is None:
        return None
    for threshold, name, desc in _THERMAL_THRESHOLDS:
        if k > threshold:
            return name, desc
    return "thermal_insulator", _THERMAL_THRESHOLDS[-1][2]


# ─── Singleton ────────────────────────────────────────────────────────────────

_kg_instance: "SimulationKnowledgeGraph | None" = None


def get_kg() -> "SimulationKnowledgeGraph":
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = SimulationKnowledgeGraph()
    return _kg_instance


# ─── Main class ───────────────────────────────────────────────────────────────

class SimulationKnowledgeGraph:

    def __init__(self):
        self._driver = None
        self._available = False
        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            self._available = True
            log.info("Knowledge graph connected to Neo4j at %s", NEO4J_URI)
        except Exception as exc:
            log.warning("Knowledge graph unavailable (Neo4j not reachable): %s", exc)
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @contextmanager
    def _session(self) -> Generator:
        if not self._available:
            raise RuntimeError("Neo4j is not available")
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()

    def _run(self, cypher: str, **params) -> list[dict]:
        """Execute a Cypher query and return rows as dicts. Returns [] on error."""
        if not self._available:
            return []
        try:
            with self._session() as s:
                result = s.run(cypher, **params)
                return [dict(r) for r in result]
        except Exception as exc:
            log.warning("KG query failed: %s | query=%s", exc, cypher[:100])
            return []

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create uniqueness constraints, full-text indexes, and vector index."""
        if not self._available:
            return
        statements = [
            "CREATE CONSTRAINT run_id_unique            IF NOT EXISTS FOR (r:Run)          REQUIRE r.run_id  IS UNIQUE",
            "CREATE CONSTRAINT material_name_unique     IF NOT EXISTS FOR (m:Material)     REQUIRE m.name    IS UNIQUE",
            "CREATE CONSTRAINT issue_code_unique        IF NOT EXISTS FOR (i:KnownIssue)   REQUIRE i.code    IS UNIQUE",
            "CREATE CONSTRAINT bcconfig_pattern_unique  IF NOT EXISTS FOR (b:BCConfig)     REQUIRE b.pattern IS UNIQUE",
            "CREATE CONSTRAINT domain_label_unique      IF NOT EXISTS FOR (d:Domain)       REQUIRE d.label   IS UNIQUE",
            "CREATE CONSTRAINT thermalclass_name_unique IF NOT EXISTS FOR (t:ThermalClass) REQUIRE t.name    IS UNIQUE",
        ]
        for stmt in statements:
            try:
                self._run(stmt)
            except Exception:
                pass

        # Neo4j 5.x native HNSW vector index for semantic Run similarity.
        # nomic-embed-text produces 768-dim vectors; cosine similarity is
        # the standard metric for sentence/document embeddings.
        try:
            self._run("""
                CREATE VECTOR INDEX run_embedding_index IF NOT EXISTS
                FOR (r:Run) ON r.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`:       768,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            log.info("Vector index run_embedding_index ensured")
        except Exception as exc:
            log.warning("Could not create vector index (non-fatal): %s", exc)

        log.info("Knowledge graph schema initialized")

    def seed_if_empty(self) -> dict:
        """
        Seed static physical knowledge if the graph is empty.
        Safe to call repeatedly — uses MERGE so no duplicates are created.
        """
        if not self._available:
            return {"seeded": False, "reason": "Neo4j unavailable"}

        from knowledge_graph.seeder import MATERIALS, KNOWN_ISSUES

        mat_count = 0
        for mat in MATERIALS:
            self._run(
                """
                MERGE (m:Material {name: $name})
                SET m.k          = $k,
                    m.rho        = $rho,
                    m.cp         = $cp,
                    m.alpha      = $alpha,
                    m.k_min      = $k_min,
                    m.k_max      = $k_max,
                    m.description   = $description,
                    m.typical_uses  = $typical_uses
                """,
                name=mat["name"],
                k=mat["k"],
                rho=mat["rho"],
                cp=mat["cp"],
                alpha=mat["alpha"],
                k_min=mat["k_range"][0],
                k_max=mat["k_range"][1],
                description=mat["description"],
                typical_uses=mat["typical_uses"],
            )
            mat_count += 1

        issue_count = 0
        for issue in KNOWN_ISSUES:
            self._run(
                """
                MERGE (i:KnownIssue {code: $code})
                SET i.severity       = $severity,
                    i.condition      = $condition,
                    i.description    = $description,
                    i.recommendation = $recommendation,
                    i.observed_in    = $observed_in
                """,
                **issue,
            )
            issue_count += 1

        # Seed ThermalClass nodes
        for threshold, name, desc in _THERMAL_THRESHOLDS:
            self._run(
                """
                MERGE (t:ThermalClass {name: $name})
                SET t.description = $description,
                    t.k_threshold = $threshold
                """,
                name=name,
                description=desc,
                threshold=float(threshold),
            )

        # Link each Material to its ThermalClass
        for mat in MATERIALS:
            tc = _thermal_class(mat["k"])
            if tc:
                self._run(
                    """
                    MATCH (m:Material {name: $mat_name})
                    MATCH (t:ThermalClass {name: $class_name})
                    MERGE (m)-[:HAS_THERMAL_CLASS]->(t)
                    """,
                    mat_name=mat["name"],
                    class_name=tc[0],
                )

        log.info("KG seeded: %d materials, %d known issues, ThermalClass nodes", mat_count, issue_count)
        return {"seeded": True, "materials": mat_count, "known_issues": issue_count}

    # ── Run management ────────────────────────────────────────────────────────

    def add_run(
        self,
        run_id: str,
        config: dict,
        results: dict,
        warnings: list[dict] | None = None,
        spawned_from: str | None = None,
    ) -> bool:
        """
        Add (or update) a Run node and attach it to material + known issues.
        Called automatically after every successful simulation.
        """
        if not self._available:
            return False
        try:
            Lx = config.get("Lx", 1.0) or 1.0
            Ly = config.get("Ly", 1.0) or 1.0
            Lz = config.get("Lz", 1.0) or 1.0

            bc_pattern = "+".join(sorted({
                bc.get("type", "unknown")
                for bc in config.get("bcs", [])
            }))
            bc_desc = _BC_DESCRIPTIONS.get(bc_pattern, f"Custom BC pattern: {bc_pattern}")

            dom_label, dom_desc = _domain_label(Lx, Ly)

            # Merge the run node — use run_id[:12] as name so Neo4j Browser
            # shows a readable caption instead of a boolean property value.
            self._run(
                """
                MERGE (r:Run {run_id: $run_id})
                SET r.name       = $name,
                    r.dim        = $dim,
                    r.status     = $status,
                    r.k          = $k,
                    r.rho        = $rho,
                    r.cp         = $cp,
                    r.nx         = $nx,
                    r.ny         = $ny,
                    r.nz         = $nz,
                    r.Lx         = $Lx,
                    r.Ly         = $Ly,
                    r.Lz         = $Lz,
                    r.bc_types   = $bc_types,
                    r.t_end      = $t_end,
                    r.dt         = $dt,
                    r.theta      = $theta,
                    r.source     = $source,
                    r.u_init     = $u_init,
                    r.t_max      = $t_max,
                    r.t_min      = $t_min,
                    r.t_mean     = $t_mean,
                    r.l2_norm    = $l2_norm,
                    r.converged  = $converged,
                    r.wall_time  = $wall_time,
                    r.n_dofs     = $n_dofs,
                    r.created_at = $created_at
                """,
                run_id=run_id,
                name=run_id[:12],
                dim=config.get("dim", 2),
                status=results.get("status", "unknown"),
                k=config.get("k"),
                rho=config.get("rho"),
                cp=config.get("cp"),
                nx=config.get("nx"),
                ny=config.get("ny"),
                nz=config.get("nz"),
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                bc_types=bc_pattern,
                t_end=config.get("t_end"),
                dt=config.get("dt"),
                theta=config.get("theta", 1.0),
                source=config.get("source", 0.0),
                u_init=config.get("u_init", 0.0),
                t_max=results.get("max_temperature"),
                t_min=results.get("min_temperature"),
                t_mean=results.get("mean_temperature"),
                l2_norm=(
                    results["convergence_history"][-1]
                    if results.get("convergence_history") else None
                ),
                converged=True,
                wall_time=results.get("wall_time"),
                n_dofs=results.get("n_dofs"),
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Create / link BCConfig node
            self._run(
                """
                MERGE (b:BCConfig {pattern: $pattern})
                SET b.description    = $description,
                    b.has_dirichlet  = $has_dirichlet,
                    b.has_neumann    = $has_neumann,
                    b.has_robin      = $has_robin,
                    b.has_source     = $has_source
                WITH b
                MATCH (r:Run {run_id: $run_id})
                MERGE (r)-[:USES_BC_CONFIG]->(b)
                """,
                pattern=bc_pattern,
                description=bc_desc,
                has_dirichlet="dirichlet" in bc_pattern,
                has_neumann="neumann" in bc_pattern,
                has_robin="robin" in bc_pattern,
                has_source=(config.get("source", 0.0) or 0.0) != 0.0,
                run_id=run_id,
            )

            # Create / link Domain node
            self._run(
                """
                MERGE (d:Domain {label: $label})
                SET d.description = $description,
                    d.Lx_ref      = $Lx,
                    d.Ly_ref      = $Ly,
                    d.char_len    = $char_len
                WITH d
                MATCH (r:Run {run_id: $run_id})
                MERGE (r)-[:ON_DOMAIN]->(d)
                """,
                label=dom_label,
                description=dom_desc,
                Lx=Lx,
                Ly=Ly,
                char_len=round(math.sqrt(Lx * Ly), 6),
                run_id=run_id,
            )

            # Link to inferred material
            material = self._infer_material(config.get("k"), config.get("rho"), config.get("cp"))
            if material:
                self._run(
                    """
                    MATCH (r:Run {run_id: $run_id})
                    MATCH (m:Material {name: $mat_name})
                    MERGE (r)-[rel:USES_MATERIAL]->(m)
                    SET rel.confidence = $conf
                    """,
                    run_id=run_id,
                    mat_name=material["name"],
                    conf=material["confidence"],
                )

            # Link to triggered known issues
            for w in (warnings or []):
                self._run(
                    """
                    MATCH (r:Run {run_id: $run_id})
                    MATCH (i:KnownIssue {code: $code})
                    MERGE (r)-[rel:TRIGGERED]->(i)
                    SET rel.detected_at = $ts
                    """,
                    run_id=run_id,
                    code=w["code"],
                    ts=datetime.now(timezone.utc).isoformat(),
                )

            # Link to parent run if spawned from a suggestion
            if spawned_from:
                self._run(
                    """
                    MATCH (child:Run {run_id: $child_id})
                    MATCH (parent:Run {run_id: $parent_id})
                    MERGE (child)-[:SPAWNED_FROM]->(parent)
                    """,
                    child_id=run_id,
                    parent_id=spawned_from,
                )

            # Compute and store embedding vector for semantic search.
            # Non-blocking: if Ollama is unavailable this is silently skipped.
            self._embed_and_store_run(run_id, config, results)

            return True
        except Exception as exc:
            log.warning("KG add_run failed for %s: %s", run_id, exc)
            return False

    def _embed_and_store_run(self, run_id: str, config: dict, results: dict) -> None:
        """
        Compute a nomic-embed-text vector for this run and store it on the
        Run node so it can be retrieved via the HNSW vector index.
        Also builds SIMILAR_TO edges to the top-k nearest neighbours.
        Silently no-ops when Ollama is not available.
        """
        try:
            from knowledge_graph.embeddings import get_embedder
            vec = get_embedder().embed_run(run_id, config, results)
            if vec:
                self._run(
                    "MATCH (r:Run {run_id: $run_id}) SET r.embedding = $vec",
                    run_id=run_id,
                    vec=vec,
                )
                log.debug("Stored embedding for run %s (%d dims)", run_id, len(vec))
                # Build SIMILAR_TO edges to nearest neighbours
                self._build_similar_to_edges(run_id, vec, k=5)
        except Exception as exc:
            log.debug("_embed_and_store_run skipped for %s: %s", run_id, exc)

    def _build_similar_to_edges(
        self, run_id: str, embedding: list[float], k: int = 5,
        min_score: float = 0.85,
    ) -> int:
        """
        Create SIMILAR_TO relationships from this run to its k nearest
        neighbours in the vector index.

        Only edges with cosine similarity ≥ min_score are created to avoid
        linking completely unrelated runs.  Existing edges are overwritten
        with the latest score.  Self-loops are excluded.

        Returns the number of edges created/updated.
        """
        try:
            rows = self._run(
                """
                CALL db.index.vector.queryNodes(
                    'run_embedding_index', $k_plus_one, $vec
                ) YIELD node AS neighbour, score
                WHERE neighbour.run_id <> $run_id
                  AND score >= $min_score
                WITH neighbour, score
                MATCH (src:Run {run_id: $run_id})
                MERGE (src)-[rel:SIMILAR_TO]->(neighbour)
                SET rel.score      = round(score, 4),
                    rel.updated_at = $ts
                RETURN count(rel) AS n_edges
                """,
                run_id=run_id,
                k_plus_one=k + 1,      # +1 because the run itself will appear
                vec=embedding,
                min_score=min_score,
                ts=datetime.now(timezone.utc).isoformat(),
            )
            n = rows[0]["n_edges"] if rows else 0
            log.debug("SIMILAR_TO edges for %s: %d created/updated", run_id, n)
            return n
        except Exception as exc:
            log.debug("_build_similar_to_edges failed for %s: %s", run_id, exc)
            return 0

    # ── Similarity search ─────────────────────────────────────────────────────

    def get_similar_runs_semantic(
        self,
        config: dict,
        results: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top-k most similar past runs using Neo4j vector similarity.

        The query config is embedded with nomic-embed-text and compared to all
        Run nodes that have an embedding stored via the HNSW index.  Returns an
        empty list if Ollama is unavailable or the index doesn't exist yet.

        Each result has an additional `similarity_score` field (0–1, higher is
        better) that the caller can use to rank alongside other signals.
        """
        try:
            from knowledge_graph.embeddings import get_embedder, run_to_text
            embedder = get_embedder()
            # Embed the *proposed* config (results may be partial/empty for pre-run)
            text = run_to_text("query", config, results or {})
            vec  = embedder.embed_text(text)
            if not vec:
                return []

            rows = self._run(
                """
                CALL db.index.vector.queryNodes(
                    'run_embedding_index', $top_k, $vec
                ) YIELD node AS r, score
                WHERE r.status = 'success'
                OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
                OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
                OPTIONAL MATCH (r)-[:ON_DOMAIN]->(d:Domain)
                OPTIONAL MATCH (m)-[:HAS_THERMAL_CLASS]->(t:ThermalClass)
                RETURN r.run_id      AS run_id,
                       r.k           AS k,
                       r.rho         AS rho,
                       r.cp          AS cp,
                       r.dim         AS dim,
                       r.nx          AS nx,
                       r.ny          AS ny,
                       r.Lx          AS Lx,
                       r.Ly          AS Ly,
                       r.t_end       AS t_end,
                       r.t_max       AS t_max,
                       r.t_min       AS t_min,
                       r.t_mean      AS t_mean,
                       r.wall_time   AS wall_time,
                       r.n_dofs      AS n_dofs,
                       r.l2_norm     AS l2_norm,
                       r.created_at  AS created_at,
                       m.name        AS material,
                       b.pattern     AS bc_pattern,
                       b.description AS bc_description,
                       d.label       AS domain_label,
                       t.name        AS thermal_class,
                       round(score, 4) AS similarity_score
                ORDER BY similarity_score DESC
                LIMIT $top_k
                """,
                top_k=top_k * 2,   # fetch extra to filter out query itself
                vec=vec,
            )
            return rows[:top_k]

        except Exception as exc:
            log.debug("Semantic similarity search failed (non-fatal): %s", exc)
            return []

    def get_similar_runs(
        self,
        config: dict,
        top_k: int = 5,
        bc_pattern: str = "",
        domain_label: str = "",
    ) -> list[dict]:
        """
        Find past runs with similar physical setup.

        Strategy (priority order):
          1. Semantic vector search via Neo4j HNSW index (when embeddings exist)
          2. Cypher parameter-distance fallback (always available)

        The Cypher fallback scoring:
          - Same BC pattern   → +1 bonus
          - Same domain class → +1 bonus
          - Sorted by (bonus DESC, |k_diff| + |nx_diff| ASC)

        Each result includes bc_pattern, domain_label, and thermal_class
        so the LLM gets full context without extra round-trips.
        """
        # ── Strategy 1: semantic vector search ───────────────────────────────
        semantic_rows = self.get_similar_runs_semantic(config, top_k=top_k)
        if semantic_rows:
            log.debug("get_similar_runs: using semantic path (%d results)", len(semantic_rows))
            return semantic_rows

        # ── Strategy 2: Cypher parameter-distance fallback ───────────────────
        log.debug("get_similar_runs: falling back to Cypher parameter search")
        k   = config.get("k", 1.0) or 1.0
        dim = config.get("dim", 2)
        nx  = config.get("nx", 20) or 20

        if not bc_pattern and config.get("bcs"):
            bc_pattern = "+".join(sorted({
                bc.get("type", "unknown") for bc in config["bcs"]
            }))

        if not domain_label:
            Lx = config.get("Lx", 1.0) or 1.0
            Ly = config.get("Ly", 1.0) or 1.0
            domain_label, _ = _domain_label(Lx, Ly)

        rows = self._run(
            """
            MATCH (r:Run)
            WHERE r.dim = $dim
              AND r.status = 'success'
              AND r.k >= $k_lo AND r.k <= $k_hi
              AND r.nx >= $nx_lo AND r.nx <= $nx_hi
            OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
            OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
            OPTIONAL MATCH (r)-[:ON_DOMAIN]->(d:Domain)
            OPTIONAL MATCH (m)-[:HAS_THERMAL_CLASS]->(t:ThermalClass)
            WITH r, m, b, d, t,
                 CASE WHEN b.pattern = $bc_pattern   THEN 1 ELSE 0 END AS bc_match,
                 CASE WHEN d.label   = $domain_label THEN 1 ELSE 0 END AS dom_match
            RETURN r.run_id      AS run_id,
                   r.k           AS k,
                   r.rho         AS rho,
                   r.cp          AS cp,
                   r.dim         AS dim,
                   r.nx          AS nx,
                   r.ny          AS ny,
                   r.Lx          AS Lx,
                   r.Ly          AS Ly,
                   r.t_end       AS t_end,
                   r.t_max       AS t_max,
                   r.t_min       AS t_min,
                   r.t_mean      AS t_mean,
                   r.wall_time   AS wall_time,
                   r.n_dofs      AS n_dofs,
                   r.l2_norm     AS l2_norm,
                   r.created_at  AS created_at,
                   m.name        AS material,
                   b.pattern     AS bc_pattern,
                   b.description AS bc_description,
                   d.label       AS domain_label,
                   t.name        AS thermal_class,
                   (bc_match + dom_match) AS relevance_score
            ORDER BY relevance_score DESC,
                     abs(r.k - $k) + abs(toFloat(r.nx) - $nx) / $nx
            LIMIT $top_k
            """,
            dim=dim,
            k_lo=k * 0.5,
            k_hi=k * 2.0,
            nx_lo=nx * 0.5,
            nx_hi=nx * 2.0,
            k=k,
            nx=float(nx),
            bc_pattern=bc_pattern,
            domain_label=domain_label,
            top_k=top_k,
        )
        return rows

    # ── BC-pattern aggregated insights ────────────────────────────────────────

    def get_bc_pattern_insights(self, bc_pattern: str) -> dict:
        """
        Return aggregated statistics for all successful runs that used a given
        boundary-condition pattern.  Gives the LLM concrete expected outcomes
        (T_max range, solve time) for a particular BC combination.
        """
        rows = self._run(
            """
            MATCH (r:Run)-[:USES_BC_CONFIG]->(b:BCConfig {pattern: $pattern})
            WHERE r.status = 'success'
            OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
            OPTIONAL MATCH (r)-[:ON_DOMAIN]->(d:Domain)
            RETURN b.pattern         AS bc_pattern,
                   b.description     AS bc_description,
                   b.has_dirichlet   AS has_dirichlet,
                   b.has_neumann     AS has_neumann,
                   b.has_robin       AS has_robin,
                   count(r)          AS run_count,
                   avg(r.t_max)      AS avg_t_max,
                   avg(r.t_min)      AS avg_t_min,
                   min(r.t_max)      AS min_t_max,
                   max(r.t_max)      AS max_t_max,
                   avg(r.wall_time)  AS avg_wall_time,
                   avg(r.l2_norm)    AS avg_l2_norm,
                   collect(DISTINCT m.name)  AS materials_used,
                   collect(DISTINCT d.label) AS domain_sizes_used
            """,
            pattern=bc_pattern,
        )
        return rows[0] if rows else {}

    # ── Domain-class aggregated insights ─────────────────────────────────────

    def get_domain_insights(self, Lx: float, Ly: float) -> dict:
        """
        Return aggregated statistics for all successful runs on the same
        domain size class as the given geometry.  Gives the LLM solve-time
        and mesh-count estimates before the user commits to a run.
        """
        label, _ = _domain_label(Lx, Ly)
        rows = self._run(
            """
            MATCH (r:Run)-[:ON_DOMAIN]->(d:Domain {label: $label})
            WHERE r.status = 'success'
            OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
            OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
            OPTIONAL MATCH (m)-[:HAS_THERMAL_CLASS]->(t:ThermalClass)
            RETURN d.label            AS domain_label,
                   d.description      AS domain_description,
                   count(r)           AS run_count,
                   avg(r.wall_time)   AS avg_wall_time,
                   min(r.wall_time)   AS min_wall_time,
                   max(r.wall_time)   AS max_wall_time,
                   avg(r.n_dofs)      AS avg_n_dofs,
                   avg(r.t_max)       AS avg_t_max,
                   collect(DISTINCT b.pattern) AS bc_patterns_used,
                   collect(DISTINCT t.name)    AS thermal_classes_used
            """,
            label=label,
        )
        return rows[0] if rows else {"domain_label": label, "run_count": 0}

    # ── ThermalClass aggregated insights ─────────────────────────────────────

    def get_thermal_class_insights(self, k: float) -> dict:
        """
        Return aggregated outcomes for all runs whose material belongs to the
        same thermal class as the given conductivity value.  Gives the LLM
        class-level behaviour across all materials with similar conductivity.
        """
        tc = _thermal_class(k)
        if tc is None:
            return {}
        class_name, class_desc = tc
        rows = self._run(
            """
            MATCH (m:Material)-[:HAS_THERMAL_CLASS]->(t:ThermalClass {name: $class_name})
            MATCH (r:Run)-[:USES_MATERIAL]->(m)
            WHERE r.status = 'success'
            OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
            OPTIONAL MATCH (r)-[:ON_DOMAIN]->(d:Domain)
            RETURN t.name              AS thermal_class,
                   $class_desc         AS thermal_class_description,
                   count(r)            AS run_count,
                   collect(DISTINCT m.name) AS materials,
                   avg(r.wall_time)    AS avg_wall_time,
                   avg(r.t_max)        AS avg_t_max,
                   min(r.t_max)        AS min_t_max,
                   max(r.t_max)        AS max_t_max,
                   collect(DISTINCT b.pattern) AS bc_patterns_tried,
                   collect(DISTINCT d.label)   AS domains_tried
            """,
            class_name=class_name,
            class_desc=class_desc,
        )
        return rows[0] if rows else {"thermal_class": class_name, "run_count": 0}

    # ── Pre-run context ───────────────────────────────────────────────────────

    def get_pre_run_context(self, config: dict) -> dict:
        """
        Return everything an agent needs before deciding to run a simulation.

        Sections returned:
          warnings               — rule violations (from rules.py)
          similar_runs           — up to 5 past runs ranked by BC+domain+k match
          inferred_material      — best-match material from KG
          past_issues_in_similar_runs — failure patterns seen in similar runs
          bc_pattern_insights    — aggregated stats for same BC combination
          domain_insights        — expected solve time / DOF count for domain size
          thermal_class_insights — class-level behaviour for same conductivity range
          kg_available           — whether KG was reachable
        """
        from knowledge_graph.rules import check_config

        k   = config.get("k")
        Lx  = config.get("Lx", 1.0) or 1.0
        Ly  = config.get("Ly", 1.0) or 1.0

        bc_pattern = "+".join(sorted({
            bc.get("type", "unknown") for bc in config.get("bcs", [])
        })) if config.get("bcs") else ""

        warnings    = check_config(config)
        similar     = self.get_similar_runs(config, top_k=5)
        material    = self._infer_material(k, config.get("rho"), config.get("cp"))
        past_issues = self._get_common_issues_for_config(config)

        bc_insights  = self.get_bc_pattern_insights(bc_pattern) if bc_pattern else {}
        dom_insights = self.get_domain_insights(Lx, Ly)
        tc_insights  = self.get_thermal_class_insights(k) if k else {}

        return {
            "warnings":                    warnings,
            "similar_runs":                similar,
            "inferred_material":           material,
            "past_issues_in_similar_runs": past_issues,
            "bc_pattern_insights":         bc_insights,
            "domain_insights":             dom_insights,
            "thermal_class_insights":      tc_insights,
            "kg_available":                self._available,
        }

    def _get_common_issues_for_config(self, config: dict) -> list[dict]:
        """Find KnownIssues triggered by similar past runs."""
        k   = config.get("k", 1.0) or 1.0
        dim = config.get("dim", 2)
        rows = self._run(
            """
            MATCH (r:Run)-[:TRIGGERED]->(i:KnownIssue)
            WHERE r.dim = $dim
              AND r.k >= $k_lo AND r.k <= $k_hi
            RETURN i.code           AS code,
                   i.severity       AS severity,
                   i.description    AS description,
                   i.recommendation AS recommendation,
                   count(r)         AS occurrence_count
            ORDER BY occurrence_count DESC
            LIMIT 5
            """,
            dim=dim,
            k_lo=k * 0.5,
            k_hi=k * 2.0,
        )
        return rows

    # ── Material inference ────────────────────────────────────────────────────

    def _infer_material(
        self, k: float | None, rho: float | None, cp: float | None
    ) -> dict | None:
        """Find the nearest material in the graph by k/rho/cp distance."""
        if k is None:
            return None

        rows = self._run(
            """
            MATCH (m:Material)
            WHERE m.k_min <= $k <= m.k_max
            RETURN m.name AS name, m.k AS k, m.rho AS rho, m.cp AS cp,
                   m.alpha AS alpha, m.description AS description,
                   m.typical_uses AS typical_uses,
                   abs(m.k - $k) AS k_diff
            ORDER BY k_diff
            LIMIT 1
            """,
            k=float(k),
        )
        if rows:
            row = rows[0]
            k_diff_frac = abs(row["k"] - k) / (row["k"] or 1.0)
            confidence  = max(0.0, 1.0 - k_diff_frac)
            return {**row, "confidence": round(confidence, 2)}

        # Fallback: nearest by absolute k distance
        rows = self._run(
            """
            MATCH (m:Material)
            RETURN m.name AS name, m.k AS k, m.rho AS rho, m.cp AS cp,
                   m.alpha AS alpha, m.description AS description,
                   m.typical_uses AS typical_uses,
                   abs(m.k - $k) AS k_diff
            ORDER BY k_diff
            LIMIT 1
            """,
            k=float(k),
        )
        if rows:
            row = rows[0]
            k_diff_frac = abs(row["k"] - k) / max(row["k"], k, 1.0)
            confidence  = max(0.0, 1.0 - k_diff_frac * 2)
            return {**row, "confidence": round(confidence, 2)}
        return None

    def get_material_info(self, name_or_k: str | float) -> dict | None:
        """
        Look up a material by name (substring, case-insensitive) or by k value.
        """
        if isinstance(name_or_k, str):
            rows = self._run(
                """
                MATCH (m:Material)
                WHERE toLower(m.name) CONTAINS toLower($name)
                RETURN m { .* } AS mat
                LIMIT 1
                """,
                name=name_or_k,
            )
            return rows[0]["mat"] if rows else None

        return self._infer_material(float(name_or_k), None, None)

    # ── Run lineage ───────────────────────────────────────────────────────────

    def get_run_lineage(self, run_id: str, depth: int = 5) -> list[dict]:
        """
        Traverse the SPAWNED_FROM chain to return the ancestry of a run.
        """
        rows = self._run(
            """
            MATCH path = (r:Run {run_id: $run_id})-[:SPAWNED_FROM*1..$depth]->(ancestor:Run)
            UNWIND nodes(path) AS n
            RETURN DISTINCT n.run_id AS run_id, n.k AS k, n.t_max AS t_max,
                   n.wall_time AS wall_time, n.created_at AS created_at
            ORDER BY n.created_at
            """,
            run_id=run_id,
            depth=depth,
        )
        return rows

    # ── Graph-wide statistics ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return high-level graph statistics."""
        if not self._available:
            return {"available": False}
        rows = self._run(
            """
            MATCH (r:Run)           RETURN count(r) AS n
            UNION ALL
            MATCH (m:Material)      RETURN count(m) AS n
            UNION ALL
            MATCH (i:KnownIssue)    RETURN count(i) AS n
            UNION ALL
            MATCH (b:BCConfig)      RETURN count(b) AS n
            UNION ALL
            MATCH (d:Domain)        RETURN count(d) AS n
            UNION ALL
            MATCH (t:ThermalClass)  RETURN count(t) AS n
            """
        )
        counts = [r["n"] for r in rows]
        return {
            "available":       True,
            "total_runs":      counts[0] if len(counts) > 0 else 0,
            "materials":       counts[1] if len(counts) > 1 else 0,
            "known_issues":    counts[2] if len(counts) > 2 else 0,
            "bc_configs":      counts[3] if len(counts) > 3 else 0,
            "domains":         counts[4] if len(counts) > 4 else 0,
            "thermal_classes": counts[5] if len(counts) > 5 else 0,
            "embedded_runs":   self._count_embedded_runs(),
        }

    def _count_embedded_runs(self) -> int:
        rows = self._run(
            "MATCH (r:Run) WHERE r.embedding IS NOT NULL RETURN count(r) AS n"
        )
        return rows[0]["n"] if rows else 0

    # ── Embedding backfill ────────────────────────────────────────────────────

    def backfill_embeddings(self, batch_size: int = 50) -> dict:
        """
        Embed all Run nodes that don't yet have an embedding vector.

        Useful after the first pull of nomic-embed-text to retroactively
        embed all existing runs.  Safe to call multiple times — skips runs
        that already have an embedding.

        Returns a summary dict: {total, already_embedded, newly_embedded, failed}.
        """
        if not self._available:
            return {"error": "Neo4j unavailable"}

        try:
            from knowledge_graph.embeddings import get_embedder, run_to_text
            embedder = get_embedder()
            if not embedder._check_available():
                return {"error": "Ollama / nomic-embed-text not available"}
        except Exception as exc:
            return {"error": str(exc)}

        # Fetch runs without embeddings
        rows = self._run(
            """
            MATCH (r:Run)
            WHERE r.embedding IS NULL
            RETURN r.run_id AS run_id,
                   r.k AS k, r.rho AS rho, r.cp AS cp,
                   r.dim AS dim, r.nx AS nx, r.ny AS ny,
                   r.Lx AS Lx, r.Ly AS Ly, r.Lz AS Lz,
                   r.t_end AS t_end, r.dt AS dt, r.source AS source,
                   r.bc_types AS bc_types, r.status AS status,
                   r.t_max AS t_max, r.t_min AS t_min, r.t_mean AS t_mean,
                   r.wall_time AS wall_time, r.n_dofs AS n_dofs
            LIMIT $limit
            """,
            limit=batch_size,
        )

        total_missing  = self._run(
            "MATCH (r:Run) WHERE r.embedding IS NULL RETURN count(r) AS n"
        )
        total_unembedded = total_missing[0]["n"] if total_missing else 0

        newly_embedded = 0
        failed         = 0

        for row in rows:
            run_id = row["run_id"]
            # Reconstruct minimal config/results dicts from stored properties
            config = {
                "k": row.get("k"), "rho": row.get("rho"), "cp": row.get("cp"),
                "dim": row.get("dim", 2), "nx": row.get("nx"), "ny": row.get("ny"),
                "Lx": row.get("Lx", 1.0), "Ly": row.get("Ly", 1.0),
                "Lz": row.get("Lz", 1.0), "t_end": row.get("t_end"),
                "dt": row.get("dt"), "source": row.get("source", 0.0),
                "bcs": [{"type": t} for t in (row.get("bc_types") or "").split("+") if t],
            }
            results = {
                "status": row.get("status", "unknown"),
                "max_temperature":  row.get("t_max"),
                "min_temperature":  row.get("t_min"),
                "mean_temperature": row.get("t_mean"),
                "wall_time":  row.get("wall_time", 0.0),
                "n_dofs":     row.get("n_dofs", 0),
            }
            vec = embedder.embed_run(run_id, config, results)
            if vec:
                self._run(
                    "MATCH (r:Run {run_id: $run_id}) SET r.embedding = $vec",
                    run_id=run_id, vec=vec,
                )
                newly_embedded += 1
            else:
                failed += 1

        already_embedded = self._count_embedded_runs() - newly_embedded
        log.info(
            "backfill_embeddings: newly_embedded=%d  failed=%d  total_unembedded=%d",
            newly_embedded, failed, total_unembedded,
        )
        return {
            "total_unembedded":  total_unembedded,
            "processed_in_batch": len(rows),
            "newly_embedded":    newly_embedded,
            "failed":            failed,
            "already_embedded":  already_embedded,
        }

    def build_all_similar_to_edges(
        self, k: int = 5, min_score: float = 0.85, batch_size: int = 50
    ) -> dict:
        """
        Backfill SIMILAR_TO edges for all embedded Run nodes.

        Iterates through all Run nodes that have an embedding and calls
        _build_similar_to_edges() for each.  Safe to run repeatedly —
        existing edges are updated with the current score.

        Returns summary: {processed, total_edges_created, skipped_no_embedding}.
        """
        if not self._available:
            return {"error": "Neo4j unavailable"}

        rows = self._run(
            """
            MATCH (r:Run)
            WHERE r.embedding IS NOT NULL
            RETURN r.run_id AS run_id, r.embedding AS embedding
            LIMIT $limit
            """,
            limit=batch_size,
        )

        total_runs = self._run(
            "MATCH (r:Run) WHERE r.embedding IS NOT NULL RETURN count(r) AS n"
        )
        total_embedded = total_runs[0]["n"] if total_runs else 0

        total_edges = 0
        processed   = 0
        offset      = 0

        while True:
            batch = self._run(
                """
                MATCH (r:Run)
                WHERE r.embedding IS NOT NULL
                RETURN r.run_id AS run_id, r.embedding AS embedding
                SKIP $skip LIMIT $limit
                """,
                skip=offset,
                limit=batch_size,
            )
            if not batch:
                break
            for row in batch:
                n = self._build_similar_to_edges(
                    row["run_id"], row["embedding"], k=k, min_score=min_score
                )
                total_edges += n
                processed   += 1
            offset += batch_size
            if len(batch) < batch_size:
                break

        # Count total SIMILAR_TO edges now in graph
        edge_count = self._run(
            "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS n"
        )
        total_in_graph = edge_count[0]["n"] if edge_count else 0

        log.info(
            "build_all_similar_to_edges: processed=%d  edges_in_graph=%d",
            processed, total_in_graph,
        )
        return {
            "total_embedded_runs": total_embedded,
            "processed":           processed,
            "total_edges_created": total_edges,
            "total_similar_to_in_graph": total_in_graph,
        }

    def close(self) -> None:
        if self._driver:
            self._driver.close()
