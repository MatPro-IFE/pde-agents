"""
Knowledge Graph Schema Migration — v1 → v2
==========================================
Applies the Phase-2 schema to all EXISTING Run and Material nodes.

New nodes created / linked:
  (:BCConfig)     — one per unique boundary-condition pattern
  (:Domain)       — one per size class (micro / component / panel / structural)
  (:ThermalClass) — one per conductivity class (high / medium / low / insulator)

New properties set on existing nodes:
  Run.name        — short display name (first 12 chars of run_id) so Neo4j
                    Browser shows a readable caption instead of "true"

Run as a one-off:
  docker compose exec agents python scripts/migrate_kg_schema_v2.py
"""

from __future__ import annotations

import math
import os
import sys

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "pde_graph_secret")

_BC_DESCRIPTIONS = {
    "dirichlet":               "Fixed temperature on all boundaries",
    "neumann":                 "Pure heat-flux / insulation on all boundaries",
    "robin":                   "Fully convective (Robin) boundaries",
    "dirichlet+neumann":       "Fixed temperature with insulated or heat-flux edges",
    "dirichlet+robin":         "Fixed temperature edges with convective cooling",
    "neumann+robin":           "Heat-flux sources with convective boundaries",
    "dirichlet+neumann+robin": "Mixed: fixed temp, heat-flux, and convective boundaries",
}

_DOMAIN_THRESHOLDS = [
    (0.015, "micro",      "Micro-scale domain  (<  1.5 cm characteristic length)"),
    (0.060, "component",  "Component-scale     (1.5–6 cm characteristic length)"),
    (0.200, "panel",      "Panel-scale         (6–20 cm characteristic length)"),
    (float("inf"), "structural", "Structural-scale (> 20 cm characteristic length)"),
]

_THERMAL_THRESHOLDS = [
    (50,  "high_conductor",   "k > 50 W/m·K — metals like copper, aluminium, silicon"),
    (10,  "medium_conductor", "10 < k ≤ 50 W/m·K — engineering steels, titanium alloys"),
    (1,   "low_conductor",    "1 < k ≤ 10 W/m·K — concrete, glass, ceramics"),
    (0,   "thermal_insulator","k ≤ 1 W/m·K — water, air, polymer foams"),
]


def domain_label(Lx: float, Ly: float) -> tuple[str, str]:
    char_len = math.sqrt(max(Lx, 1e-12) * max(Ly, 1e-12))
    for threshold, label, desc in _DOMAIN_THRESHOLDS:
        if char_len < threshold:
            return label, desc
    return "structural", _DOMAIN_THRESHOLDS[-1][2]


def thermal_class(k: float | None) -> tuple[str, str] | None:
    if k is None:
        return None
    for threshold, name, desc in _THERMAL_THRESHOLDS:
        if k > threshold:
            return name, desc
    return "thermal_insulator", _THERMAL_THRESHOLDS[-1][2]


def run(driver) -> None:
    with driver.session() as s:

        # ── 0. Constraints for new node types ─────────────────────────────────
        print("Creating constraints …")
        for stmt in [
            "CREATE CONSTRAINT bcconfig_pattern_unique IF NOT EXISTS FOR (b:BCConfig)    REQUIRE b.pattern IS UNIQUE",
            "CREATE CONSTRAINT domain_label_unique     IF NOT EXISTS FOR (d:Domain)      REQUIRE d.label   IS UNIQUE",
            "CREATE CONSTRAINT thermalclass_name_unique IF NOT EXISTS FOR (t:ThermalClass) REQUIRE t.name  IS UNIQUE",
        ]:
            s.run(stmt)

        # ── 1. Fix Run.name so Neo4j Browser shows run_id instead of "true" ──
        print("Setting Run.name …")
        res = s.run(
            "MATCH (r:Run) WHERE r.name IS NULL "
            "SET r.name = left(r.run_id, 12) "
            "RETURN count(r) AS n"
        )
        print(f"  Updated {res.single()['n']} Run nodes")

        # ── 2. ThermalClass nodes ─────────────────────────────────────────────
        print("Creating ThermalClass nodes …")
        for threshold, name, desc in _THERMAL_THRESHOLDS:
            s.run(
                "MERGE (t:ThermalClass {name: $name}) SET t.description = $desc, t.k_threshold = $thr",
                name=name, desc=desc, thr=float(threshold),
            )

        # ── 3. Link Materials → ThermalClass ─────────────────────────────────
        print("Linking Material → ThermalClass …")
        mats = [dict(r) for r in s.run("MATCH (m:Material) RETURN m.name AS name, m.k AS k")]
        linked = 0
        for mat in mats:
            tc = thermal_class(mat["k"])
            if tc:
                s.run(
                    """
                    MATCH (m:Material {name: $mat_name})
                    MATCH (t:ThermalClass {name: $class_name})
                    MERGE (m)-[:HAS_THERMAL_CLASS]->(t)
                    """,
                    mat_name=mat["name"], class_name=tc[0],
                )
                linked += 1
        print(f"  Linked {linked} materials")

        # ── 4. Read all Run nodes ─────────────────────────────────────────────
        print("Reading Run nodes …")
        runs = [dict(r) for r in s.run(
            "MATCH (r:Run) RETURN r.run_id AS run_id, r.bc_types AS bc_types, "
            "r.Lx AS Lx, r.Ly AS Ly, r.source AS source"
        )]
        print(f"  Found {len(runs)} Run nodes")

        # ── 5. BCConfig nodes + USES_BC_CONFIG edges ──────────────────────────
        print("Creating BCConfig nodes and linking runs …")
        bc_runs = 0
        for r in runs:
            pattern = r.get("bc_types") or "unknown"
            desc = _BC_DESCRIPTIONS.get(pattern, f"Custom BC pattern: {pattern}")
            s.run(
                """
                MERGE (b:BCConfig {pattern: $pattern})
                SET b.description   = $desc,
                    b.has_dirichlet = $hd,
                    b.has_neumann   = $hn,
                    b.has_robin     = $hr,
                    b.has_source    = $hs
                WITH b
                MATCH (run:Run {run_id: $run_id})
                MERGE (run)-[:USES_BC_CONFIG]->(b)
                """,
                pattern=pattern,
                desc=desc,
                hd="dirichlet" in pattern,
                hn="neumann" in pattern,
                hr="robin" in pattern,
                hs=bool(r.get("source") and r["source"] != 0),
                run_id=r["run_id"],
            )
            bc_runs += 1
        print(f"  Processed {bc_runs} runs → BCConfig")

        # ── 6. Domain nodes + ON_DOMAIN edges ────────────────────────────────
        print("Creating Domain nodes and linking runs …")
        dom_runs = 0
        for r in runs:
            Lx = float(r.get("Lx") or 1.0)
            Ly = float(r.get("Ly") or 1.0)
            lbl, desc = domain_label(Lx, Ly)
            char_len = round(math.sqrt(Lx * Ly), 6)
            s.run(
                """
                MERGE (d:Domain {label: $label})
                SET d.description = $desc,
                    d.Lx_ref      = $Lx,
                    d.Ly_ref      = $Ly,
                    d.char_len    = $char_len
                WITH d
                MATCH (run:Run {run_id: $run_id})
                MERGE (run)-[:ON_DOMAIN]->(d)
                """,
                label=lbl, desc=desc, Lx=Lx, Ly=Ly, char_len=char_len,
                run_id=r["run_id"],
            )
            dom_runs += 1
        print(f"  Processed {dom_runs} runs → Domain")

        # ── 7. Summary ────────────────────────────────────────────────────────
        print("\n=== Post-migration node counts ===")
        for label in ["Run", "Material", "KnownIssue", "BCConfig", "Domain", "ThermalClass"]:
            res = s.run(f"MATCH (n:{label}) RETURN count(n) AS n")
            print(f"  {label:16s}: {res.single()['n']}")

        print("\n=== Relationship types ===")
        rel_res = s.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS n ORDER BY n DESC"
        )
        for row in rel_res:
            print(f"  {row['rel']:25s}: {row['n']}")


def main():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j driver not found — install it inside the container")
        sys.exit(1)

    print(f"Connecting to {NEO4J_URI} …")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected.\n")
        run(driver)
        print("\nMigration complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
