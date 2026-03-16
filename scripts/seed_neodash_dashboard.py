#!/usr/bin/env python3
"""
Seed a pre-built NeoDash dashboard into Neo4j.

NeoDash stores dashboards as (:NeoDashDashboard) nodes in Neo4j.
The dashboard title must match the `standaloneDashboardName` env var
set in docker-compose.yml (default: "PDE Agents KG").

Usage:
    docker exec pde-agents python3 /app/scripts/seed_neodash_dashboard.py
    # or from the repo root:
    python3 scripts/seed_neodash_dashboard.py
"""

import json
import os
import sys
from datetime import datetime, timezone

# ── Neo4j connection ──────────────────────────────────────────────────────────

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "pde_graph_secret")
DASH_TITLE = os.getenv("NEODASH_TITLE",  "PDE Agents KG")

# ── Dashboard definition ──────────────────────────────────────────────────────

def _report(rid, title, query, rtype, x, y, w, h, selection=None, settings=None):
    return {
        "id":        str(rid),
        "title":     title,
        "query":     query.strip(),
        "type":      rtype,
        "width":     w,
        "height":    h,
        "x":         x,
        "y":         y,
        "selection": selection or {},
        "settings":  settings  or {},
    }


DASHBOARD = {
    "title":   DASH_TITLE,
    "version": "2.4",
    "settings": {
        "pagenumber":       0,
        "editable":         True,
        "fullscreenEnabled": False,
        "parameters":       {},
        "theme":            "dark",
    },
    "pages": [

        # ── Page 1: Overview ──────────────────────────────────────────────────
        {
            "title": "📊 Overview",
            "reports": [
                _report("kpi_runs", "Total Runs",
                    "MATCH (r:Run) RETURN count(r) AS value",
                    "value", 0, 0, 3, 2,
                    settings={"fontSize": 56, "color": "#00b4d8", "textAlign": "center"}),

                _report("kpi_ok", "Successful",
                    "MATCH (r:Run {status:'success'}) RETURN count(r) AS value",
                    "value", 3, 0, 3, 2,
                    settings={"fontSize": 56, "color": "#6fd672", "textAlign": "center"}),

                _report("kpi_failed", "Failed",
                    "MATCH (r:Run {status:'failed'}) RETURN count(r) AS value",
                    "value", 6, 0, 3, 2,
                    settings={"fontSize": 56, "color": "#ff6b6b", "textAlign": "center"}),

                _report("kpi_emb", "Embedded Runs",
                    "MATCH (r:Run) WHERE r.embedding IS NOT NULL RETURN count(r) AS value",
                    "value", 9, 0, 3, 2,
                    settings={"fontSize": 56, "color": "#b48eff", "textAlign": "center"}),

                _report("tbl_recent", "Recent Simulation Runs",
                    """
MATCH (r:Run)
OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
RETURN r.run_id          AS run_id,
       r.status          AS status,
       r.dim             AS dim,
       round(r.t_max, 1) AS t_max_K,
       round(r.wall_time,2) AS wall_s,
       m.name            AS material,
       b.pattern         AS bc_pattern
ORDER BY r.created_at DESC LIMIT 30
                    """,
                    "table", 0, 2, 9, 6,
                    settings={"compact": True}),

                _report("pie_status", "Status Distribution",
                    "MATCH (r:Run) RETURN r.status AS status, count(r) AS count ORDER BY count DESC",
                    "pie", 9, 2, 3, 3,
                    selection={"index": "status", "value": "count", "key": "status"}),

                _report("bar_dim", "Runs by Dimension",
                    "MATCH (r:Run) RETURN toString(r.dim)+'D' AS dim, count(r) AS runs",
                    "bar", 9, 5, 3, 3,
                    selection={"index": "dim", "value": "runs", "key": "dim"},
                    settings={"barValues": True, "colors": "neodash"}),
            ],
        },

        # ── Page 2: Knowledge Graph ───────────────────────────────────────────
        {
            "title": "🧠 Knowledge Graph",
            "reports": [
                _report("graph_mat", "Run → Material Connections  (latest 40 runs)",
                    """
MATCH (r:Run)-[rel:USES_MATERIAL]->(m:Material)
WITH r, rel, m ORDER BY r.created_at DESC LIMIT 40
RETURN r, rel, m
                    """,
                    "graph", 0, 0, 8, 7,
                    selection={"Run": "run_id", "Material": "name"},
                    settings={"nodeColorScheme": "category10",
                              "defaultNodeSize": "large",
                              "relationshipParticles": False}),

                _report("bar_mat", "Runs per Material",
                    """
MATCH (r:Run)-[:USES_MATERIAL]->(m:Material)
RETURN m.name AS material, count(r) AS runs
ORDER BY runs DESC
                    """,
                    "bar", 8, 0, 4, 7,
                    selection={"index": "material", "value": "runs", "key": "material"},
                    settings={"barValues": True, "layout": "horizontal"}),

                _report("graph_sim", "Semantic Similarity Network  (top 30 SIMILAR_TO edges)",
                    """
MATCH (r:Run)-[rel:SIMILAR_TO]->(s:Run)
RETURN r, rel, s
ORDER BY rel.score DESC LIMIT 30
                    """,
                    "graph", 0, 7, 12, 6,
                    selection={"Run": "run_id"},
                    settings={"nodeColorScheme": "category10",
                              "defaultNodeSize": "medium"}),
            ],
        },

        # ── Page 3: BC & Domain Analysis ─────────────────────────────────────
        {
            "title": "🔲 BC & Domain",
            "reports": [
                _report("tbl_bc", "BC Pattern Outcomes",
                    """
MATCH (r:Run)-[:USES_BC_CONFIG]->(b:BCConfig)
WHERE r.status = 'success'
RETURN b.pattern          AS bc_pattern,
       count(r)           AS runs,
       round(avg(r.t_max),1)     AS avg_t_max_K,
       round(min(r.t_max),1)     AS min_t_max_K,
       round(max(r.t_max),1)     AS max_t_max_K,
       round(avg(r.wall_time),2) AS avg_wall_s
ORDER BY runs DESC
                    """,
                    "table", 0, 0, 6, 5,
                    settings={"compact": True}),

                _report("pie_domain", "Domain Scale Distribution",
                    """
MATCH (r:Run)-[:ON_DOMAIN]->(d:Domain)
RETURN d.label AS domain, count(r) AS runs
ORDER BY runs DESC
                    """,
                    "pie", 6, 0, 3, 5,
                    selection={"index": "domain", "value": "runs", "key": "domain"}),

                _report("bar_issues", "KnownIssue Trigger Frequency",
                    """
MATCH (r:Run)-[:TRIGGERED]->(i:KnownIssue)
RETURN i.code AS issue, count(r) AS occurrences
ORDER BY occurrences DESC
                    """,
                    "bar", 9, 0, 3, 5,
                    selection={"index": "issue", "value": "occurrences", "key": "issue"},
                    settings={"barValues": True, "layout": "horizontal"}),

                _report("tbl_top", "Top Runs by T_max",
                    """
MATCH (r:Run {status:'success'})
OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
OPTIONAL MATCH (r)-[:ON_DOMAIN]->(d:Domain)
RETURN r.run_id           AS run_id,
       round(r.t_max,1)   AS t_max_K,
       round(r.t_min,1)   AS t_min_K,
       m.name             AS material,
       b.pattern          AS bc_pattern,
       d.label            AS domain,
       r.dim              AS dim,
       round(r.wall_time,3) AS wall_s
ORDER BY r.t_max DESC LIMIT 20
                    """,
                    "table", 0, 5, 12, 5,
                    settings={"compact": True}),
            ],
        },

        # ── Page 4: Physics References ────────────────────────────────────────
        {
            "title": "📚 Physics References",
            "reports": [
                _report("tbl_refs", "All Physics References",
                    """
MATCH (ref:Reference)
OPTIONAL MATCH (m:Material)-[:HAS_REFERENCE]->(ref)
OPTIONAL MATCH (b:BCConfig)-[:HAS_REFERENCE]->(ref)
RETURN ref.type    AS type,
       ref.subject AS subject,
       ref.text    AS text,
       ref.source  AS source,
       ref.url     AS url,
       collect(DISTINCT m.name)    AS materials,
       collect(DISTINCT b.pattern) AS bc_patterns
ORDER BY ref.type, ref.subject
                    """,
                    "table", 0, 0, 12, 12,
                    settings={"compact": False}),
            ],
        },
    ],
}


# ── Seed into Neo4j ───────────────────────────────────────────────────────────

def seed(uri: str, user: str, password: str, title: str, dashboard: dict) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("ERROR: neo4j driver not installed. Run: pip install neo4j")
        sys.exit(1)

    content = json.dumps(dashboard, ensure_ascii=False)
    now     = datetime.now(timezone.utc).isoformat()

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run(
            """
            MERGE (n:NeoDashDashboard {title: $title})
            SET   n.content  = $content,
                  n.date     = $date,
                  n.user     = 'admin',
                  n.version  = '2.4'
            RETURN n.title AS title
            """,
            title=title, content=content, date=now,
        )
        record = result.single()
        print(f"✓ Dashboard seeded: '{record['title']}'")
        print(f"  Content length:   {len(content):,} chars")
        print(f"  Pages:            {len(dashboard['pages'])}")
        total_reports = sum(len(p['reports']) for p in dashboard['pages'])
        print(f"  Total reports:    {total_reports}")
    driver.close()


if __name__ == "__main__":
    print(f"Seeding NeoDash dashboard '{DASH_TITLE}' → {NEO4J_URI} ...")
    seed(NEO4J_URI, NEO4J_USER, NEO4J_PASS, DASH_TITLE, DASHBOARD)
    print("\nDone. Refresh NeoDash in your browser to load the dashboard.")
