"""
Agent-2: Analytics & Visualization Agent

Responsibilities:
  1. Analyze completed simulation results (stats, convergence, heat flux)
  2. Cross-reference different parametric variations
  3. Suggest configuration changes for new runs → communicates with Agent-1
  4. Set up visualization of simulation results (reports, plots metadata)
"""

from __future__ import annotations

import os
from typing import Optional

from agents.base_agent import BaseAgent
from tools.knowledge_tools import query_knowledge_graph
from tools.analytics_tools import (
    analyze_run,
    compare_runs,
    compare_study,
    export_summary_report,
    get_steady_state_time,
    list_runs_for_analysis,
    suggest_next_run,
)

ANALYTICS_MODEL = os.getenv("ANALYTICS_AGENT_MODEL", "llama4:scout")


ANALYTICS_SYSTEM_PROMPT = """You are Agent-2: the Analytics & Visualization Agent in a multi-agent PDE solving ecosystem.

Your role is to analyze simulation results, identify patterns and insights, compare parametric
variations, and suggest improvements to be implemented by the Simulation Agent.

## Your capabilities (tools):
- query_knowledge_graph: Look up material properties, find similar past runs by k+dim,
  or get a run's lineage. Use BEFORE analysis when the user asks about a material.
- list_runs_for_analysis: ALWAYS call this FIRST when the user asks about past runs
  without specifying run IDs. Returns available run_ids and summary stats.
- analyze_run: Deep statistical analysis of a completed run
- compare_runs: Side-by-side comparison of multiple runs
- compare_study: Full analysis of a parametric sweep study
- suggest_next_run: Generate smart configuration suggestions based on results
- get_steady_state_time: Estimate when a solution reaches steady state
- export_summary_report: Create a JSON summary report across runs

## Workflow for "analyze my runs" type questions:
1. Call list_runs_for_analysis({}) to discover which runs exist
2. Select the relevant run_ids based on the user's question
3. Call analyze_run / compare_runs / suggest_next_run as appropriate
4. Synthesize a clear, quantitative answer

## Analysis framework:
When analyzing heat equation results, focus on:

1. **Temperature field quality**:
   - Uniformity index (1=perfectly uniform, 0=maximum non-uniformity)
   - Peak temperature (risk of material failure?)
   - Temperature gradient (stress indicator)

2. **Convergence behavior**:
   - Is the L2 norm monotonically decreasing? (should be for stable schemes)
   - Has steady state been reached?
   - Rate of convergence (exponential vs algebraic)

3. **Physical plausibility**:
   - Does T_max/T_min match boundary conditions?
   - Conservation check: energy in = energy out?
   - Thermal diffusivity: α = k/(ρ c_p)
   - Characteristic time: τ = L²/α (should match simulation time)

4. **Parametric sensitivity**:
   - How sensitive is T_max to k? (should scale linearly for simple BCs)
   - Does increasing mesh resolution change the answer? (mesh convergence)
   - Is the time step small enough? (try halving dt, compare results)

## Visualization strategy:
Describe what plots should be generated:
- Temperature field contour plots (2D colormap or 3D volume rendering)
- Convergence history (L2 norm vs time)
- Parametric sensitivity charts (metric vs parameter)
- Comparison bar charts for cross-run analysis

## Communication with Simulation Agent:
When you suggest a new run, provide:
1. The exact configuration changes needed
2. Clear scientific rationale
3. Expected outcome (what should improve)
4. Priority level (1=urgent, 5=optional)

Be quantitative, scientific, and actionable in your analysis.
"""


class AnalyticsAgent(BaseAgent):
    """Agent-2: Analyzes simulation results and suggests improvements."""

    system_prompt = ANALYTICS_SYSTEM_PROMPT
    tools = [
        query_knowledge_graph,
        list_runs_for_analysis,  # discover available runs first
        analyze_run,
        compare_runs,
        compare_study,
        suggest_next_run,
        get_steady_state_time,
        export_summary_report,
    ]
    model_name = ANALYTICS_MODEL
    agent_name = "analytics"
    max_iterations = 12

    def analyze(self, run_id: str) -> dict:
        """Perform full analysis on a single completed run."""
        task = f"""Perform a comprehensive analysis of simulation run '{run_id}'.

Please:
1. Analyze the run statistics (temperature field, convergence)
2. Check if steady state was reached
3. Assess the quality of the solution
4. Identify any issues or anomalies
5. Suggest what the next simulation should be to improve or extend this result
6. Specify the visualization that would best communicate these results

Be specific and quantitative in your analysis."""

        return self.run(task, context={"run_id": run_id})

    def compare_and_suggest(self, run_ids: list[str], goal: str = "") -> dict:
        """Compare multiple runs and generate a suggestion for the next best run."""
        import json
        task = f"""Compare these simulation runs and suggest the optimal next configuration.

Run IDs: {json.dumps(run_ids)}
Goal: {goal or 'Maximize thermal uniformity while minimizing computation time'}

Steps:
1. Compare all runs side-by-side
2. Identify which configuration performed best and why
3. Analyze the sensitivity to each varied parameter
4. Suggest the single best next configuration to run
5. Explain the scientific reasoning behind your suggestion"""

        return self.run(task, context={"run_ids": run_ids})

    def study_analysis(self, study_id: str) -> dict:
        """Full analysis of a parametric study."""
        task = f"""Analyze the complete parametric study '{study_id}'.

Steps:
1. Load and compare all runs in the study
2. Compute sensitivity of results to the swept parameter
3. Identify the optimal parameter value
4. Describe the trend (monotonic? non-linear? has optimal point?)
5. Suggest follow-up studies to further optimize
6. Create a summary report

Provide a publication-quality analysis with clear conclusions."""

        return self.run(task, context={"study_id": study_id})

    def generate_visualization_spec(self, run_ids: list[str]) -> dict:
        """Generate a visualization specification for the dashboard."""
        import json
        task = f"""Generate a visualization specification for runs: {json.dumps(run_ids)}

For each run, specify what plots to create:
1. 2D/3D temperature field visualization
2. Convergence history plot
3. Boundary condition overlay
4. Comparison charts if multiple runs provided

Return a structured JSON spec that the dashboard can use.
Each plot should have: type, data_source, axes labels, color scheme, title."""

        return self.run(task, context={"run_ids": run_ids})
