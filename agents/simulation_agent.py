"""
Agent-1: Simulation Agent

Responsibilities:
  1. Set up simulation configurations from natural language descriptions
  2. Launch FEM heat equation simulations (2D/3D via FEniCSx)
  3. Monitor running simulations
  4. Debug failed or slow simulations and auto-retry with fixes
  5. Run parametric sweeps

KG integration modes (controlled via constructor flags):
  - disable_kg=False (default "KG On"):  Mandatory KG-first workflow
  - disable_kg=True  ("KG Off"):         KG tools removed entirely
  - smart_kg=True    ("KG Smart"):       Warm-start + lazy/conditional KG
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from agents.base_agent import BaseAgent
from tools.simulation_tools import (
    debug_simulation,
    get_run_status,
    list_recent_runs,
    modify_config,
    run_parametric_sweep,
    run_simulation,
    validate_config,
)
from tools.knowledge_tools import check_config_warnings, query_knowledge_graph

log = logging.getLogger(__name__)

SIM_MODEL = os.getenv("SIMULATION_AGENT_MODEL", "qwen3-coder-next")


# ─── Base system prompt (shared preamble) ──────────────────────────────────────

_PROMPT_PREAMBLE = """You are Agent-1: the Simulation Agent in a multi-agent PDE solving ecosystem.

Your role is to set up, launch, monitor, and debug finite element method (FEM) simulations
of partial differential equations, starting with the heat equation.

## Your capabilities (tools):
- run_simulation: Launch a heat equation FEM run with FEniCSx
- validate_config: Check a config for errors before running
- modify_config: Patch simulation configuration fields
- debug_simulation: Analyze failed runs and propose fixes
- list_recent_runs: Show recent simulation runs
- get_run_status: Check status of a specific run
- run_parametric_sweep: Run the same simulation across a range of parameter values

## Heat equation configuration guide:
The heat equation: ρ c_p ∂u/∂t - ∇·(k∇u) = f

Key config fields:
  dim: 2 (2D on [0,1]²) or 3 (3D on [0,1]³)
  nx, ny, nz: mesh resolution (nodes per edge)
  k: thermal conductivity [W/(m·K)]
  rho: density [kg/m³]
  cp: specific heat capacity [J/(kg·K)]
  source: body heat generation [W/m³]
  t_end: final simulation time [s]
  dt: time step size [s]
  theta: 1.0=Backward Euler (stable), 0.5=Crank-Nicolson (accurate)
  bcs: list of boundary conditions:
    {"type": "dirichlet", "value": T_value, "location": "left|right|top|bottom|front|back"}
    {"type": "neumann",   "value": flux,    "location": "..."}
    {"type": "robin",     "alpha": h_coef,  "u_inf": T_ambient, "location": "..."}

## Important rules:
- ALWAYS validate before running
- ALWAYS set u_init to a value consistent with the Dirichlet BCs to avoid numerical
  overshoot. If left wall=273K and right wall=373K, set u_init=323.0 (midpoint) or
  match the lower BC value. Never leave u_init at 0.0 when BCs are high temperatures.
- If dt is too large relative to mesh size and diffusivity, reduce it
- For stability: dt ≤ h²/(2α) where α=k/(ρ c_p) is thermal diffusivity (for explicit)
- Backward Euler (theta=1) is unconditionally stable - prefer it
- Start with coarse meshes (nx=ny=32) then refine if needed
- Always include all boundary conditions (all 4 sides in 2D, all 6 faces in 3D)

Be concise, systematic, and always explain what you're doing.
"""

# ─── KG-mode specific prompt suffixes ──────────────────────────────────────────

_KG_ON_SUFFIX = """
## Your capabilities (knowledge tools):
- query_knowledge_graph: Look up material properties by name (e.g. "steel", "copper"),
  find similar runs by k+dim, or get the lineage of a run.
- check_config_warnings: Check a config against known failure patterns and similar past runs.

## MANDATORY Workflow — follow these steps IN ORDER:
1. Parse the task. If a material is named, call query_knowledge_graph ONCE for its
   properties (k, rho, cp). Use the returned values — do NOT substitute your own.
   If the material is not found, use your best knowledge for its thermal properties.
2. Build the config JSON from the task description + retrieved properties.
3. Call check_config_warnings ONCE with the config JSON.
4. Call run_simulation with the config JSON. YOU MUST ALWAYS CALL run_simulation.
   Never just output a config — you must actually run the simulation.
5. If simulation failed: call debug_simulation, fix, retry run_simulation.
6. Report the final result including the run_id.

CRITICAL: You MUST call run_simulation. A task is NOT complete until run_simulation
has been called and returned a run_id. Never stop at just building a config.
"""

_KG_OFF_SUFFIX = """
NOTE: Knowledge-graph tools (check_config_warnings, query_knowledge_graph) are NOT
available in this session. Skip straight to validate_config then run_simulation.

## Workflow:
1. Parse the user's task description into a simulation config
2. Validate the config
3. Launch the simulation
4. If failed: debug, apply fixes, and retry (up to 3 times)
5. Report results to the user
"""

_KG_SMART_SUFFIX = """
## Your capabilities (knowledge tools — use ONLY when needed):
- check_config_warnings: Check config against known failure patterns and find similar past runs.
  DO NOT call this before every run — only use it if:
    (a) a simulation FAILED and you need to diagnose why, or
    (b) the user mentions a material by name and you are unsure of its properties.
- query_knowledge_graph: Look up material properties or find similar past runs.
  Use ONLY when you need specific material data you don't already know.

## Workflow (streamlined — minimize KG round-trips):
1. Parse the user's task description into a simulation config
2. If similar past runs are provided below, use them as reference for parameters
3. Validate the config (validate_config)
4. Launch the simulation (run_simulation)
5. ONLY IF the simulation fails: call check_config_warnings for diagnosis, then fix and retry
6. Report results to the user

IMPORTANT: Go directly to validate_config → run_simulation unless you have a specific
reason to query the knowledge graph. Speed and directness are priorities.
"""


def _get_warm_start_context(task_description: str) -> str:
    """Query KG for similar past successful runs and format as prompt context.

    This runs BEFORE the agent loop, injecting relevant past configurations
    as few-shot examples so the LLM can leverage run history without needing
    to make tool calls during reasoning.

    Returns empty string if KG is unavailable or no similar runs exist.
    """
    try:
        from knowledge_graph.graph import get_kg
        from knowledge_graph.embeddings import get_embedder
        kg = get_kg()
        if not kg or not kg.available:
            return ""

        embedder = get_embedder()
        vec = embedder.embed_text(task_description)
        if not vec:
            return ""

        rows = kg._run(
            """
            CALL db.index.vector.queryNodes(
                'run_embedding_index', $top_k, $vec
            ) YIELD node AS r, score
            WHERE r.status = 'success' AND score >= $min_score
            OPTIONAL MATCH (r)-[:USES_MATERIAL]->(m:Material)
            OPTIONAL MATCH (r)-[:USES_BC_CONFIG]->(b:BCConfig)
            RETURN r.run_id    AS run_id,
                   r.k         AS k,
                   r.rho       AS rho,
                   r.cp        AS cp,
                   r.dim       AS dim,
                   r.nx        AS nx,
                   r.ny        AS ny,
                   r.nz        AS nz,
                   r.t_end     AS t_end,
                   r.dt        AS dt,
                   r.theta     AS theta,
                   r.source    AS source,
                   r.t_max     AS t_max,
                   r.t_min     AS t_min,
                   r.wall_time AS wall_time,
                   r.n_dofs    AS n_dofs,
                   r.bc_types  AS bc_types,
                   m.name      AS material,
                   m.k         AS mat_k,
                   m.rho       AS mat_rho,
                   m.cp        AS mat_cp,
                   b.pattern   AS bc_pattern,
                   round(score, 4) AS similarity
            ORDER BY score DESC
            LIMIT $top_k
            """,
            top_k=3,
            vec=vec,
            min_score=0.70,
        )

        if not rows:
            return ""

        lines = ["\n## Similar past successful runs (from Knowledge Graph):"]
        lines.append("Use these as reference when choosing parameters.\n")
        for i, r in enumerate(rows, 1):
            mat_info = ""
            if r.get("material"):
                mat_info = (f"  material: {r['material']} "
                           f"(k={r.get('mat_k')}, rho={r.get('mat_rho')}, cp={r.get('mat_cp')})")
            lines.append(f"### Reference run {i} (similarity={r.get('similarity', '?')}):")
            lines.append(f"  dim={r.get('dim')}, nx={r.get('nx')}, ny={r.get('ny')}, "
                        f"k={r.get('k')}, rho={r.get('rho')}, cp={r.get('cp')}")
            lines.append(f"  t_end={r.get('t_end')}, dt={r.get('dt')}, "
                        f"theta={r.get('theta')}, source={r.get('source')}")
            lines.append(f"  BCs: {r.get('bc_types', 'unknown')}")
            if mat_info:
                lines.append(mat_info)
            lines.append(f"  Result: T_max={r.get('t_max')}, T_min={r.get('t_min')}, "
                        f"wall_time={r.get('wall_time')}s, n_dofs={r.get('n_dofs')}")
            lines.append("")

        return "\n".join(lines)

    except Exception as exc:
        log.debug("Warm-start KG query failed (non-fatal): %s", exc)
        return ""


SIMULATION_SYSTEM_PROMPT = _PROMPT_PREAMBLE + _KG_ON_SUFFIX


class SimulationAgent(BaseAgent):
    """Agent-1: Sets up, runs, monitors, and debugs FEM simulations."""

    system_prompt = SIMULATION_SYSTEM_PROMPT
    tools = [
        check_config_warnings,
        query_knowledge_graph,
        run_simulation,
        validate_config,
        modify_config,
        debug_simulation,
        list_recent_runs,
        get_run_status,
        run_parametric_sweep,
    ]
    model_name = SIM_MODEL
    agent_name = "simulation"
    max_iterations = 25

    def __init__(self, disable_kg: bool = False, smart_kg: bool = False,
                 **kwargs):
        if smart_kg:
            self.system_prompt = _PROMPT_PREAMBLE + _KG_SMART_SUFFIX
            self._smart_kg = True
        elif disable_kg:
            self.tools = [t for t in type(self).tools
                          if t.name not in ("check_config_warnings",
                                            "query_knowledge_graph")]
            self.system_prompt = _PROMPT_PREAMBLE + _KG_OFF_SUFFIX
            self._smart_kg = False
        else:
            self._smart_kg = False

        super().__init__(
            model_name=kwargs.get("model_name"),
            max_iterations=kwargs.get("max_iterations"),
            temperature=kwargs.get("temperature"),
            ollama_base_url=kwargs.get("ollama_base_url"),
        )

    _nudge_count: int = 0
    _MAX_NUDGES: int = 3

    def _router(self, state) -> str:
        """Override router: nudge the LLM up to 3 times if it hasn't
        called run_simulation yet."""
        from langchain_core.messages import AIMessage

        last = state["messages"][-1]
        iteration = state["iteration"]
        max_iter = state["max_iterations"]

        if iteration >= max_iter:
            self._nudge_count = 0
            return "finish"

        if isinstance(last, AIMessage) and last.tool_calls:
            return "act"

        context = state.get("context", {})
        if (not context.get("run_id")
                and self._nudge_count < self._MAX_NUDGES
                and iteration < max_iter - 2):
            from langchain_core.messages import HumanMessage
            self._nudge_count += 1
            nudge = HumanMessage(
                content="You have not called run_simulation yet. "
                        "Please call run_simulation now with your config JSON. "
                        "Do NOT output text — call the run_simulation tool."
            )
            state["messages"].append(nudge)
            return "retry"

        self._nudge_count = 0
        return "finish"

    def _build_graph(self):
        """Override to add a retry self-loop back to reason."""
        from langgraph.graph import StateGraph, END
        from agents.base_agent import AgentState

        g = StateGraph(AgentState)

        g.add_node("reason", self._reason_node)
        g.add_node("act",    self.tool_node)
        g.add_node("finish", self._finish_node)

        g.set_entry_point("reason")

        g.add_conditional_edges(
            "reason",
            self._router,
            {
                "act":    "act",
                "finish": "finish",
                "retry":  "reason",
            },
        )
        g.add_edge("act", "reason")
        g.add_edge("finish", END)

        return g.compile()

    _AUTO_RETRY: int = 2

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        """Override to inject KG warm-start context and auto-retry on
        stochastic failures (no run_id produced).

        Only retries when the agent used less than 60% of its iteration
        budget — this targets stochastic early-termination (LLM emits text
        instead of a tool call) while avoiding a costly double-run for
        systematic budget-exhaustion failures.
        """
        for attempt in range(self._AUTO_RETRY):
            self._nudge_count = 0
            if self._smart_kg:
                warm_ctx = _get_warm_start_context(task)
                if warm_ctx:
                    self.system_prompt = _PROMPT_PREAMBLE + _KG_SMART_SUFFIX + warm_ctx
                    log.debug("Smart KG: injected %d chars of warm-start context",
                              len(warm_ctx))
            result = super().run(task, context)
            if result.get("run_id"):
                return result
            iters_used = result.get("iterations", 0)
            budget = self.max_iterations
            if attempt < self._AUTO_RETRY - 1 and iters_used < budget * 0.6:
                log.info("No run_id after %d/%d iterations (early exit) "
                         "— auto-retry %d/%d",
                         iters_used, budget, attempt + 1,
                         self._AUTO_RETRY - 1)
            else:
                break
        return result

    def setup_and_run(
        self,
        description: str,
        dim: int = 2,
        auto_retry: bool = True,
    ) -> dict:
        """
        High-level method: set up a simulation from a description and run it.

        Args:
            description: Natural language description of the simulation.
            dim:         Spatial dimension (2 or 3).
            auto_retry:  Whether to auto-debug and retry on failure.

        Returns:
            Agent result dict with run_id and final status.
        """
        task = f"""Set up and run a {dim}D heat equation simulation.

Description: {description}

Please:
1. Create an appropriate simulation configuration for this scenario
2. Validate the config
3. Run the simulation
4. Report the results including temperature range and wall time

If the run fails, debug it and retry with fixes."""

        return self.run(task)

    def debug_run(self, run_id: str) -> dict:
        """Debug a specific failed or suspicious simulation run."""
        task = f"""Debug simulation run '{run_id}'.

1. Check the run status and retrieve any error information
2. Analyze the configuration for potential issues
3. Diagnose the most likely cause of failure
4. Propose specific configuration fixes
5. If you can, create a corrected configuration and run it

Provide a clear diagnosis and action plan."""

        return self.run(task, context={"run_id": run_id})

    def run_sweep(
        self,
        parameter: str,
        values: list,
        base_config_description: str,
    ) -> dict:
        """Run a parametric sweep over a given parameter."""
        import json
        task = f"""Run a parametric sweep varying '{parameter}' over values {values}.

Base configuration description: {base_config_description}

Steps:
1. Create a base configuration appropriate for this description
2. Run the parametric sweep
3. Summarize which parameter value gave the best results"""

        return self.run(task)
