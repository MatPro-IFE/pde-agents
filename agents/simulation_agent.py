"""
Agent-1: Simulation Agent

Responsibilities:
  1. Set up simulation configurations from natural language descriptions
  2. Launch FEM heat equation simulations (2D/3D via FEniCSx)
  3. Monitor running simulations
  4. Debug failed or slow simulations and auto-retry with fixes
  5. Run parametric sweeps
"""

from __future__ import annotations

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

SIM_MODEL = os.getenv("SIMULATION_AGENT_MODEL", "qwen2.5-coder:32b")


SIMULATION_SYSTEM_PROMPT = """You are Agent-1: the Simulation Agent in a multi-agent PDE solving ecosystem.

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

## Your capabilities (knowledge tools):
- check_config_warnings: ALWAYS call this BEFORE running any simulation. Checks the config
  against known failure patterns AND finds similar past runs with their outcomes.
- query_knowledge_graph: Look up material properties by name (e.g. "steel", "copper"),
  find similar runs by k+dim, or get the lineage of a run.

## Workflow:
1. Parse the user's task description into a simulation config
2. Call check_config_warnings to validate against known issues and see past similar runs
3. Validate the config (check for errors, estimate resources)
4. Launch the simulation
4. Monitor until completion
5. If failed: debug, apply fixes, and retry (up to 3 times)
6. Report results to the user

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
    max_iterations = 15

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
