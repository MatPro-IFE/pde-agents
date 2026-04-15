#!/usr/bin/env python3
"""
Seed the Neo4j knowledge graph with the fictional material "Novidium".

This script:
  1. Triggers the standard KG seeder (which now includes Novidium in MATERIALS)
  2. Seeds the Reference nodes (which now include Novidium reference chunks)
  3. Verifies that the Material node and linked References exist

Run inside the agents container:
    docker compose exec agents python /app/evaluation/ablation/seed_novidium.py

Or from host (if PYTHONPATH includes the project root and Neo4j is reachable):
    python evaluation/ablation/seed_novidium.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/app")


def main() -> None:
    from knowledge_graph.graph import get_kg

    kg = get_kg()
    if not kg or not kg.available:
        print("ERROR: Neo4j is not available. Is the neo4j container running?")
        sys.exit(1)

    print("1/3  Seeding materials (including Novidium)...")
    mat_result = kg.seed_if_empty()
    print(f"     → {mat_result}")

    print("2/3  Seeding reference nodes (including Novidium references)...")
    ref_result = kg.seed_references()
    print(f"     → {ref_result}")

    print("3/3  Verifying Novidium exists in the KG...")
    info = kg.get_material_info("novidium")
    if info:
        print(f"     ✓ Material found: {info['name']}")
        print(f"       k={info.get('k')} W/m·K, ρ={info.get('rho')} kg/m³, "
              f"cp={info.get('cp')} J/kg·K")
        print(f"       Description: {info.get('description', '')[:120]}...")
    else:
        print("     ✗ Novidium Material node NOT found — seeding may have failed.")
        sys.exit(1)

    refs = kg._run(
        """
        MATCH (m:Material)-[:HAS_REFERENCE]->(r:Reference)
        WHERE toLower(m.name) CONTAINS 'novidium'
        RETURN r.ref_id AS ref_id, r.subject AS subject
        """,
    )
    if refs:
        print(f"     ✓ {len(refs)} Reference(s) linked to Novidium:")
        for r in refs:
            print(f"       - {r['ref_id']}: {r['subject']}")
    else:
        print("     ⚠ No Reference nodes linked to Novidium. "
              "Re-running seed_references to force linking...")
        kg.seed_references()
        refs2 = kg._run(
            """
            MATCH (m:Material)-[:HAS_REFERENCE]->(r:Reference)
            WHERE toLower(m.name) CONTAINS 'novidium'
            RETURN r.ref_id AS ref_id, r.subject AS subject
            """,
        )
        if refs2:
            print(f"     ✓ {len(refs2)} Reference(s) now linked.")
        else:
            print("     ✗ Still no references. Check references.py material_names field.")

    print("\nDone. Novidium is ready for the ablation experiment.")


if __name__ == "__main__":
    main()
