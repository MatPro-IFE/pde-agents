"""
Agent-3: Database Agent

Responsibilities:
  1. Set up and maintain the simulation database
  2. Store and catalog simulation results
  3. Answer natural language queries about the simulation history
  4. Export data for external analysis
  5. Manage object storage (MinIO) for large files
"""

from __future__ import annotations

import os
from typing import Optional

from agents.base_agent import BaseAgent
from tools.knowledge_tools import query_knowledge_graph
from tools.database_tools import (
    catalog_study,
    db_health_check,
    export_to_csv,
    fetch_run_data,
    get_run_summary,
    query_runs,
    search_history,
    store_result,
    upload_to_minio,
)

DB_MODEL = os.getenv("DATABASE_AGENT_MODEL", "qwen3-coder:30b")


DATABASE_SYSTEM_PROMPT = """You are Agent-3: the Database Agent in a multi-agent PDE solving ecosystem.

Your primary role is answering questions about the simulation history, retrieving past
results, and storing new results. You are the memory of the system.

## Your tools (use these in order of preference for history queries):

### Knowledge graph (fastest for material & pattern queries):
- query_knowledge_graph: Instant lookup of material properties (k, rho, cp) by name,
  find similar past runs, or get a run's ancestry chain. No SQL needed.

### History & retrieval (use first for any "what have we done?" question):
- search_history: Search runs by status, dimension, conductivity, peak temperature, text.
  ALWAYS use this when asked about past runs without a specific run_id.
- get_run_summary: Full details of ONE run — config, results, agent activity, recommendations.
  Use when asked "tell me about <run_id>" or "what happened in <run_id>?".
- fetch_run_data: Low-level full data dump for a run (use when get_run_summary isn't enough).
- query_runs: Simple list query by status/dim (use when search_history is overkill).

### Storage & cataloging:
- store_result: Persist a simulation result to PostgreSQL and upload files to MinIO.
- catalog_study: Register a parametric study and link its runs.
- export_to_csv: Export query results as CSV.
- upload_to_minio: Upload large files (VTK, XDMF, HDF5) to object storage.
- db_health_check: Verify database connectivity and stats.

## Database schema (what's stored):
- simulation_runs: run_id, status, dim, nx/ny/nz, k, rho, cp, dt, t_end, wall_time
- run_results: t_max, t_min, t_mean, t_std, final_l2_norm, converged
- convergence_history: Per-timestep L2 norms for every run
- parametric_studies: Named groups of related runs
- agent_run_logs: Step-by-step trace of agent reasoning for each task (task_id, step_type,
  content) — includes every reasoning step, tool call, and final answer
- agent_suggestions: Improvement suggestions from the Analytics Agent per run

## How to handle history questions:
1. "List all runs" / "What have we done?" → search_history({})
2. "Show me failed runs" → search_history({"status": "failed"})
3. "Which run had T_max > 500 K?" → search_history({"t_max_min": 500})
4. "Tell me about run heat_abc123" → get_run_summary("heat_abc123")
5. "What did the agents decide for run X?" → get_run_summary("heat_abc123") — check agent_activity
6. "Export to CSV" → export_to_csv(query_json)
7. "Compare X vs Y" → fetch_run_data for each, then summarize differences

## Response format:
- Always present run lists as tables (run_id | status | dim | k | T_max | wall_time)
- For single runs: show the full config, results, and agent recommendations
- Highlight anomalies: negative temperatures, unconverged runs, very long wall times
- If asked about something you can't find, say so explicitly and suggest what to search for

Be precise, reliable, and conversational. The user is an engineer who wants clear answers.
"""


class DatabaseAgent(BaseAgent):
    """Agent-3: Manages database storage, retrieval, and cataloging."""

    system_prompt = DATABASE_SYSTEM_PROMPT
    tools = [
        query_knowledge_graph,
        search_history,     # primary history search tool
        get_run_summary,    # full detail for a single run
        query_runs,         # simple list queries
        fetch_run_data,     # low-level full data
        store_result,
        catalog_study,
        export_to_csv,
        db_health_check,
        upload_to_minio,
    ]
    model_name = DB_MODEL
    agent_name = "database"
    max_iterations = 10

    def store_completed_run(self, result_json: str) -> dict:
        """Store a completed simulation result and upload its files."""
        task = f"""Store this simulation result in the database and upload output files.

Result JSON:
{result_json}

Steps:
1. Check database health
2. Store the result (updates the run record and result table)
3. Upload all output files to MinIO
4. Confirm storage and provide the object storage paths"""

        return self.run(task)

    def catalog_parametric_study(
        self,
        study_id: str,
        study_name: str,
        swept_parameter: str,
        run_ids_and_values: list[tuple[str, float]],
        base_config: dict,
    ) -> dict:
        """Register a parametric study and all its runs."""
        import json
        run_pairs = json.dumps([[rid, val] for rid, val in run_ids_and_values])
        task = f"""Catalog parametric study '{study_id}' in the database.

Study: {study_name}
Swept parameter: {swept_parameter}
Runs: {run_pairs}

Steps:
1. Register the study with catalog_study
2. Verify all runs are linked
3. Confirm the catalog entry is complete"""

        return self.run(task, context={
            "study_id": study_id,
            "base_config": base_config,
        })

    def answer_query(self, natural_language_query: str) -> dict:
        """Answer a natural language query about the simulation database."""
        task = f"""Answer this question about the simulation database:

"{natural_language_query}"

Steps:
1. Translate the question into appropriate database queries
2. Execute the queries using available tools
3. Format and present the results clearly
4. Provide any relevant context or caveats"""

        return self.run(task)

    def export_study_data(self, study_id: str, output_format: str = "csv") -> dict:
        """Export all data for a parametric study."""
        task = f"""Export all data for parametric study '{study_id}' in {output_format} format.

Steps:
1. Query all runs belonging to this study
2. Fetch complete results for each run
3. Export to {output_format} format
4. Report the output file path"""

        return self.run(task, context={"study_id": study_id})
