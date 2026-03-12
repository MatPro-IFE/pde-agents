"""
Simulation Knowledge Graph — Phase 1

Stores simulation runs, material knowledge, and learned patterns in Neo4j.
Degrades gracefully if Neo4j is unavailable.
"""

from knowledge_graph.graph import SimulationKnowledgeGraph, get_kg

__all__ = ["SimulationKnowledgeGraph", "get_kg"]
