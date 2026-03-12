"""
LangChain tools for the Simulation Knowledge Graph.

Provides two tools:
  query_knowledge_graph  — General-purpose KG query for agents
  check_config_warnings  — Pre-run validation; returns warnings + similar runs
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

log = logging.getLogger(__name__)


def _kg():
    """Lazy-load the KG singleton so import errors don't crash agents."""
    try:
        from knowledge_graph.graph import get_kg
        return get_kg()
    except Exception as exc:
        log.warning("KG unavailable: %s", exc)
        return None


# ─── Tool 1: General-purpose knowledge graph query ────────────────────────────

@tool
def query_knowledge_graph(
    question: str,
    material: str = "",
    k: float = 0.0,
    dim: int = 0,
    run_id: str = "",
    bc_pattern: str = "",
    domain_label: str = "",
) -> str:
    """
    Query the simulation knowledge graph for physical knowledge, similar past runs,
    and learned patterns. Use this tool to:

      1. Look up material properties
         → set material="copper" or material="steel"
      2. Find similar past simulations before running a new one
         → set k=50.0, dim=2 (or 3)
      3. Narrow similarity search by boundary conditions or domain size
         → set bc_pattern="dirichlet+robin", domain_label="component"
      4. Get aggregated stats for a BC combination
         → set bc_pattern="dirichlet+neumann" (without k/dim)
      5. Get aggregated stats for a domain size class
         → set domain_label="micro" | "component" | "panel" | "structural"
      6. Get the ancestry / lineage of a specific run
         → set run_id="heat_abc123"
      7. Ask general questions about materials or PDE physics
         → set question="What material has k≈200?"

    Args:
        question:     Natural language question or intent
        material:     Optional material name (e.g. "aluminium", "copper")
        k:            Optional thermal conductivity — find nearest material or similar runs
        dim:          Optional simulation dimension (2 or 3) for run similarity search
        run_id:       Optional run ID to get lineage/ancestry
        bc_pattern:   Optional BC pattern filter, e.g. "dirichlet+robin",
                      "neumann", "dirichlet+neumann+robin"
        domain_label: Optional domain size class filter: "micro", "component",
                      "panel", or "structural"

    Returns:
        JSON string with the answer and supporting data
    """
    kg = _kg()
    if kg is None or not kg.available:
        return json.dumps({
            "available": False,
            "message": "Knowledge graph is not available right now. Neo4j may be starting up.",
        })

    response: dict = {"question": question, "kg_available": True}

    # ── Material lookup ────────────────────────────────────────────────────────
    if material.strip():
        info = kg.get_material_info(material.strip())
        if info:
            tc = kg.get_thermal_class_insights(info.get("k", 0))
            response["material"] = info
            response["thermal_class_insights"] = tc
            response["summary"] = (
                f"{info['name']}: k={info.get('k')} W/m·K, "
                f"ρ={info.get('rho')} kg/m³, cp={info.get('cp')} J/kg·K. "
                f"{info.get('description', '')} "
                f"Typical uses: {info.get('typical_uses', '')}. "
                f"Thermal class: {tc.get('thermal_class', 'unknown')} "
                f"({tc.get('run_count', 0)} past runs in this class, "
                f"avg T_max={_fmt(tc.get('avg_t_max'))} K)."
            )
        else:
            response["material"] = None
            response["summary"] = f"No material named '{material}' found in the knowledge graph."
        return json.dumps(response, default=str)

    # ── BC-pattern stats (without k/dim — aggregate view) ────────────────────
    if bc_pattern.strip() and k == 0.0 and dim == 0:
        insights = kg.get_bc_pattern_insights(bc_pattern.strip())
        response["bc_pattern_insights"] = insights
        if insights:
            response["summary"] = (
                f"BC pattern '{bc_pattern}': {insights.get('bc_description', '')}. "
                f"{insights.get('run_count', 0)} successful runs. "
                f"T_max range: {_fmt(insights.get('min_t_max'))}–{_fmt(insights.get('max_t_max'))} K "
                f"(avg {_fmt(insights.get('avg_t_max'))} K). "
                f"Avg solve time: {_fmt(insights.get('avg_wall_time'))} s. "
                f"Materials tried: {', '.join(insights.get('materials_used', []))}. "
                f"Domain sizes tried: {', '.join(insights.get('domain_sizes_used', []))}."
            )
        else:
            response["summary"] = f"No successful runs found for BC pattern '{bc_pattern}'."
        return json.dumps(response, default=str)

    # ── Domain size stats (without k/dim — aggregate view) ───────────────────
    if domain_label.strip() and k == 0.0 and dim == 0:
        insights = kg.get_domain_insights(
            {"micro": 0.005, "component": 0.04, "panel": 0.1, "structural": 0.5}.get(domain_label, 0.1),
            {"micro": 0.005, "component": 0.02, "panel": 0.05, "structural": 0.2}.get(domain_label, 0.05),
        )
        response["domain_insights"] = insights
        if insights and insights.get("run_count", 0) > 0:
            response["summary"] = (
                f"Domain class '{domain_label}': {insights.get('domain_description', '')}. "
                f"{insights.get('run_count', 0)} successful runs. "
                f"Avg solve time: {_fmt(insights.get('avg_wall_time'))} s "
                f"(range {_fmt(insights.get('min_wall_time'))}–{_fmt(insights.get('max_wall_time'))} s). "
                f"Avg DOFs: {_fmt(insights.get('avg_n_dofs'), fmt='.0f')}. "
                f"BC patterns tried: {', '.join(insights.get('bc_patterns_used', []))}."
            )
        else:
            response["summary"] = f"No successful runs found for domain class '{domain_label}'."
        return json.dumps(response, default=str)

    # ── Nearest material by k value ────────────────────────────────────────────
    if k > 0 and dim == 0:
        info = kg.get_material_info(k)
        if info:
            tc = kg.get_thermal_class_insights(k)
            response["inferred_material"] = info
            response["thermal_class_insights"] = tc
            response["summary"] = (
                f"k={k} W/m·K is closest to {info['name']} "
                f"(k={info.get('k')}, confidence={info.get('confidence', 0):.0%}). "
                f"{info.get('description', '')}. "
                f"Thermal class '{tc.get('thermal_class', 'unknown')}' has "
                f"{tc.get('run_count', 0)} past runs, avg T_max={_fmt(tc.get('avg_t_max'))} K."
            )
        return json.dumps(response, default=str)

    # ── Similar run search ─────────────────────────────────────────────────────
    if k > 0 and dim in (2, 3):
        similar = kg.get_similar_runs(
            {"k": k, "dim": dim},
            top_k=5,
            bc_pattern=bc_pattern.strip(),
            domain_label=domain_label.strip(),
        )
        response["similar_runs"] = similar

        # Also attach aggregate insights so LLM has richer context
        if bc_pattern.strip():
            response["bc_pattern_insights"] = kg.get_bc_pattern_insights(bc_pattern.strip())
        if domain_label.strip():
            response["domain_insights"] = kg.get_domain_insights(
                {"micro": 0.005, "component": 0.04, "panel": 0.1, "structural": 0.5}.get(domain_label, 0.1),
                {"micro": 0.005, "component": 0.02, "panel": 0.05, "structural": 0.2}.get(domain_label, 0.05),
            )
        response["thermal_class_insights"] = kg.get_thermal_class_insights(k)

        if similar:
            best = similar[0]
            bc_note = f", BC='{best.get('bc_pattern', '?')}'" if best.get("bc_pattern") else ""
            dom_note = f", domain='{best.get('domain_label', '?')}'" if best.get("domain_label") else ""
            tc_note = f", class='{best.get('thermal_class', '?')}'" if best.get("thermal_class") else ""
            response["summary"] = (
                f"Found {len(similar)} similar {dim}D run(s) with k≈{k}{bc_note}{dom_note}{tc_note}. "
                f"Top match: {best.get('run_id')} — "
                f"T_max={_fmt(best.get('t_max'))} K, "
                f"T_min={_fmt(best.get('t_min'))} K, "
                f"wall={_fmt(best.get('wall_time'))} s, "
                f"material={best.get('material', 'unknown')}, "
                f"relevance_score={best.get('relevance_score', 0)}/2."
            )
        else:
            response["summary"] = (
                f"No similar {dim}D runs found with k≈{k}"
                + (f", bc_pattern='{bc_pattern}'" if bc_pattern else "")
                + (f", domain='{domain_label}'" if domain_label else "")
                + "."
            )
        return json.dumps(response, default=str)

    # ── Run lineage ────────────────────────────────────────────────────────────
    if run_id.strip():
        lineage = kg.get_run_lineage(run_id.strip())
        response["lineage"] = lineage
        response["summary"] = (
            f"Run {run_id} has {len(lineage)} ancestor(s) in the knowledge graph."
            if lineage else
            f"No lineage found for run {run_id} — it may be a root run."
        )
        return json.dumps(response, default=str)

    # ── General question — return graph stats as context ──────────────────────
    stats = kg.stats()
    response["graph_stats"] = stats
    response["summary"] = (
        f"Knowledge graph contains {stats.get('total_runs', 0)} simulation runs across "
        f"{stats.get('bc_configs', 0)} BC patterns, "
        f"{stats.get('domains', 0)} domain size classes, "
        f"{stats.get('thermal_classes', 0)} thermal classes, "
        f"{stats.get('materials', 0)} materials, and "
        f"{stats.get('known_issues', 0)} documented failure patterns. "
        "To get specific results, set material=, k=, dim=, bc_pattern=, domain_label=, or run_id=."
    )
    return json.dumps(response, default=str)


def _fmt(val, fmt: str = ".1f") -> str:
    """Format a numeric value safely, returning '?' for None."""
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "?"


# ─── Tool 2: Pre-run config validation ────────────────────────────────────────

@tool
def check_config_warnings(config_json: str) -> str:
    """
    Validate a simulation configuration against known failure patterns and
    find similar successful past runs BEFORE running a simulation.

    Always call this tool when the user provides a new simulation configuration
    to check for likely problems and what similar runs achieved.

    Args:
        config_json: JSON string of the proposed simulation configuration
                     (same format as run_simulation config_json)

    Returns:
        JSON string with:
          - warnings:                   list of triggered rule violations
          - similar_runs:               up to 5 past runs ranked by BC+domain+k match
          - inferred_material:          best-guess material for the given k/rho/cp
          - past_issues_in_similar_runs: failure patterns seen in similar runs
          - bc_pattern_insights:        aggregate stats for same BC combination
          - domain_insights:            expected solve time / DOF count for domain size
          - thermal_class_insights:     class-level behaviour for same conductivity range
          - recommendation:             overall pass/warning/caution assessment
    """
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid config JSON: {exc}"})

    kg = _kg()

    from knowledge_graph.rules import check_config
    warnings = check_config(config)

    if kg and kg.available:
        ctx = kg.get_pre_run_context(config)
        ctx["warnings"] = warnings
    else:
        ctx = {
            "warnings": warnings,
            "similar_runs": [],
            "inferred_material": None,
            "past_issues_in_similar_runs": [],
            "bc_pattern_insights": {},
            "domain_insights": {},
            "thermal_class_insights": {},
            "kg_available": False,
        }

    # Severity summary
    high_count   = sum(1 for w in warnings if w["severity"] == "high")
    medium_count = sum(1 for w in warnings if w["severity"] == "medium")

    if high_count > 0:
        recommendation = (
            f"CAUTION: {high_count} high-severity issue(s) detected. "
            "Review warnings before running — results may be incorrect."
        )
    elif medium_count > 0:
        recommendation = (
            f"WARNING: {medium_count} medium-severity issue(s). "
            "Run will likely succeed but results may be inaccurate."
        )
    elif warnings:
        recommendation = "Minor issues detected. Safe to proceed but review recommendations."
    else:
        recommendation = "Config looks valid. No known issues detected."

    # Enrich recommendation with BC and domain context from KG
    similar = ctx.get("similar_runs", [])
    if similar:
        t_max_vals = [r["t_max"] for r in similar if r.get("t_max") is not None]
        if t_max_vals:
            recommendation += (
                f" Similar runs achieved T_max "
                f"{min(t_max_vals):.0f}–{max(t_max_vals):.0f} K."
            )

    bc_ins = ctx.get("bc_pattern_insights", {})
    if bc_ins and bc_ins.get("run_count", 0) > 0:
        recommendation += (
            f" This BC pattern ('{bc_ins.get('bc_pattern')}') has been used in "
            f"{bc_ins['run_count']} past runs with avg T_max={_fmt(bc_ins.get('avg_t_max'))} K "
            f"and avg solve time={_fmt(bc_ins.get('avg_wall_time'))} s."
        )

    dom_ins = ctx.get("domain_insights", {})
    if dom_ins and dom_ins.get("run_count", 0) > 0:
        recommendation += (
            f" On '{dom_ins.get('domain_label')}'-scale domains, "
            f"expect ~{_fmt(dom_ins.get('avg_wall_time'))} s solve time "
            f"and ~{_fmt(dom_ins.get('avg_n_dofs'), fmt='.0f')} DOFs."
        )

    ctx["recommendation"] = recommendation
    return json.dumps(ctx, default=str)
