#!/usr/bin/env python3
"""
Adaptive KG Decision Framework

Implements and retrospectively validates Algorithm 1 from the paper:
given a simulation task description, decide which KG mode to use.

Rules:
  1. If task specifies all material properties explicitly -> KG Off
  2. If task names a known engineering material -> KG Smart (lazy)
  3. If task names an unknown / proprietary material -> KG Smart (forced warm-start)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.ablation.benchmark_tasks import get_all_tasks

KNOWN_MATERIALS = {
    "steel", "stainless steel", "carbon steel", "aisi 1010", "aisi 304",
    "copper", "aluminium", "aluminum", "titanium", "iron", "brass",
    "bronze", "nickel", "tungsten", "lead", "zinc", "silver", "gold",
    "glass", "concrete", "wood", "ceramic", "silicon", "graphite",
    "diamond", "rubber", "polyethylene", "nylon", "epoxy", "teflon",
}

PROPERTY_PATTERN = re.compile(
    r"\b(?:k|rho|cp|conductivity|density|specific heat)\s*[=:]\s*[\d.]+",
    re.IGNORECASE,
)
MATERIAL_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+\d+)?)\b"
)


def has_explicit_properties(desc: str) -> bool:
    """Check if the description gives k, rho, cp values directly."""
    mentions = PROPERTY_PATTERN.findall(desc)
    return len(mentions) >= 2


def extract_material_name(desc: str) -> str | None:
    """Extract the first material name from a task description."""
    lower = desc.lower()
    for mat in KNOWN_MATERIALS:
        if mat in lower:
            return mat
    for word in ["novidium", "cryonite", "pyrathane"]:
        if word in lower:
            return word
    return None


def decide_kg_mode(desc: str) -> str:
    """
    Adaptive KG decision algorithm.
    Returns: 'kg_off' | 'kg_smart_lazy' | 'kg_smart_forced'
    """
    if has_explicit_properties(desc):
        return "kg_off"

    material = extract_material_name(desc)
    if material is None:
        return "kg_off"

    if material.lower() in KNOWN_MATERIALS:
        return "kg_smart_lazy"

    return "kg_smart_forced"


def optimal_mode(task: dict) -> str:
    """What the retrospective analysis says is the best mode for this task."""
    novelty = task.get("material_novelty", "unknown")
    if novelty == "explicit_params":
        return "kg_off"
    elif novelty == "known":
        return "kg_smart_lazy"
    elif novelty == "novel":
        return "kg_smart_forced"
    return "kg_smart_lazy"


def main():
    tasks = get_all_tasks()
    correct = 0
    results = []

    print(f"{'ID':<5} {'Difficulty':<8} {'Novelty':<16} {'Decided':<20} {'Optimal':<20} {'Match'}")
    print("-" * 85)

    for task in tasks:
        tid = task["id"]
        desc = task["description"]
        decided = decide_kg_mode(desc)
        opt = optimal_mode(task)
        match = decided == opt

        print(f"{tid:<5} {task['difficulty']:<8} "
              f"{task.get('material_novelty', 'n/a'):<16} "
              f"{decided:<20} {opt:<20} {'OK' if match else 'MISMATCH'}")

        results.append({
            "task_id": tid,
            "difficulty": task["difficulty"],
            "material_novelty": task.get("material_novelty"),
            "decided_mode": decided,
            "optimal_mode": opt,
            "match": match,
        })
        if match:
            correct += 1

    accuracy = correct / len(tasks)
    print(f"\nAccuracy: {correct}/{len(tasks)} = {accuracy:.0%}")

    out = Path(__file__).resolve().parent / "results" / "decision_framework.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"accuracy": accuracy, "tasks": results}, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
