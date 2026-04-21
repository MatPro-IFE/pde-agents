#!/usr/bin/env python3
"""
KG Snapshot Manager — backup, restore, and reset the Neo4j knowledge graph.

Used to ensure ablation experiments run against a frozen KG state.

Usage:
    # Snapshot current KG state
    python evaluation/kg_snapshot.py save --name baseline_v2

    # Restore from snapshot (wipes current KG first)
    python evaluation/kg_snapshot.py restore --name baseline_v2

    # Wipe KG completely (for clean growth experiments)
    python evaluation/kg_snapshot.py reset

    # List available snapshots
    python evaluation/kg_snapshot.py list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "pde_neo4j_secret")

SNAPSHOT_DIR = Path(__file__).resolve().parent / "kg_snapshots"


def _get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _export_all(driver) -> dict:
    """Export all nodes and relationships as JSON-serializable dicts."""
    nodes = []
    rels = []
    with driver.session() as s:
        result = s.run(
            "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props"
        )
        for record in result:
            nodes.append({
                "id": record["id"],
                "labels": record["labels"],
                "props": dict(record["props"]),
            })

        result = s.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, "
            "type(r) AS type, properties(r) AS props"
        )
        for record in result:
            rels.append({
                "src": record["src"],
                "dst": record["dst"],
                "type": record["type"],
                "props": dict(record["props"]),
            })

    return {"nodes": nodes, "relationships": rels}


def _wipe(driver):
    """Delete all nodes and relationships."""
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("  KG wiped.")


def _import_all(driver, data: dict):
    """Restore nodes and relationships from exported data."""
    old_to_new = {}
    with driver.session() as s:
        for node in data["nodes"]:
            labels = ":".join(node["labels"])
            result = s.run(
                f"CREATE (n:{labels} $props) RETURN elementId(n) AS new_id",
                props=node["props"],
            )
            new_id = result.single()["new_id"]
            old_to_new[node["id"]] = new_id

        for rel in data["relationships"]:
            src_new = old_to_new.get(rel["src"])
            dst_new = old_to_new.get(rel["dst"])
            if src_new is not None and dst_new is not None:
                s.run(
                    f"MATCH (a), (b) WHERE elementId(a) = $src AND elementId(b) = $dst "
                    f"CREATE (a)-[r:{rel['type']}]->(b) SET r = $props",
                    src=src_new,
                    dst=dst_new,
                    props=rel["props"],
                )

    print(f"  Restored {len(data['nodes'])} nodes, {len(data['relationships'])} relationships.")


def _rebuild_indexes(driver):
    """Recreate HNSW and other indexes after a restore."""
    with driver.session() as s:
        try:
            s.run(
                "CREATE VECTOR INDEX run_embedding IF NOT EXISTS "
                "FOR (r:Run) ON (r.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
            )
            print("  HNSW index recreated.")
        except Exception as e:
            print(f"  Warning: could not create HNSW index: {e}")


def save_snapshot(name: str):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    driver = _get_driver()
    try:
        data = _export_all(driver)
        data["metadata"] = {
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "node_count": len(data["nodes"]),
            "rel_count": len(data["relationships"]),
        }
        out = SNAPSHOT_DIR / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, default=str)
        print(f"Snapshot saved: {out}")
        print(f"  Nodes: {data['metadata']['node_count']}")
        print(f"  Relationships: {data['metadata']['rel_count']}")
    finally:
        driver.close()


def restore_snapshot(name: str):
    snap_file = SNAPSHOT_DIR / f"{name}.json"
    if not snap_file.exists():
        print(f"Error: snapshot '{name}' not found at {snap_file}")
        sys.exit(1)

    with open(snap_file) as f:
        data = json.load(f)

    print(f"Restoring snapshot: {name}")
    print(f"  Nodes: {data['metadata']['node_count']}, Rels: {data['metadata']['rel_count']}")

    driver = _get_driver()
    try:
        _wipe(driver)
        _import_all(driver, data)
        _rebuild_indexes(driver)
        print("Restore complete.")
    finally:
        driver.close()


def reset_kg():
    driver = _get_driver()
    try:
        _wipe(driver)
        _rebuild_indexes(driver)
        print("KG reset to empty state.")
    finally:
        driver.close()


def list_snapshots():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SNAPSHOT_DIR.glob("*.json"))
    if not files:
        print("No snapshots found.")
        return
    print(f"{'Name':<30s} {'Nodes':>8s} {'Rels':>8s} {'Created':>20s}")
    print("-" * 70)
    for f in files:
        try:
            meta = json.load(open(f))["metadata"]
            print(f"{meta['name']:<30s} {meta['node_count']:>8d} {meta['rel_count']:>8d} {meta['created_at'][:19]:>20s}")
        except Exception:
            print(f"{f.stem:<30s} {'?':>8s} {'?':>8s} {'?':>20s}")


def kg_stats():
    """Print current KG node/relationship counts."""
    driver = _get_driver()
    try:
        with driver.session() as s:
            n_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            n_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            print(f"Current KG: {n_nodes} nodes, {n_rels} relationships")

            # Per-label counts
            result = s.run(
                "MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c ORDER BY c DESC"
            )
            for r in result:
                print(f"  {r['l']:<20s}: {r['c']:>6d}")
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Snapshot Manager")
    sub = parser.add_subparsers(dest="command")

    p_save = sub.add_parser("save", help="Save current KG to a snapshot")
    p_save.add_argument("--name", required=True)

    p_restore = sub.add_parser("restore", help="Restore KG from a snapshot")
    p_restore.add_argument("--name", required=True)

    sub.add_parser("reset", help="Wipe KG to empty state")
    sub.add_parser("list", help="List available snapshots")
    sub.add_parser("stats", help="Print current KG stats")

    args = parser.parse_args()
    if args.command == "save":
        save_snapshot(args.name)
    elif args.command == "restore":
        restore_snapshot(args.name)
    elif args.command == "reset":
        reset_kg()
    elif args.command == "list":
        list_snapshots()
    elif args.command == "stats":
        kg_stats()
    else:
        parser.print_help()
