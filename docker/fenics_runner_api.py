"""
Lightweight FastAPI server running inside the FEniCSx container.

This server receives simulation jobs from the Simulation Agent,
executes them using the FEniCSx heat equation solver,
and returns results.

Endpoints:
  POST /run     - execute a simulation job
  GET  /status  - server health
  GET  /results/{run_id} - fetch result metadata
"""

from __future__ import annotations

import sys
import json
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add workspace to path
sys.path.insert(0, "/workspace")

from simulations.solvers.heat_equation import HeatConfig, HeatEquationSolver

app = FastAPI(title="FEniCSx Runner API", version="1.0.0")


class SimulationJob(BaseModel):
    """Mirrors HeatConfig fields."""
    dim: int = 2
    nx: int = 32
    ny: int = 32
    nz: int = 16
    rho: float = 1.0
    cp: float = 1.0
    k: float = 1.0
    source: float = 0.0
    bcs: list = []
    u_init: float = 0.0
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.01
    theta: float = 1.0
    element_degree: int = 1
    output_dir: str = "/workspace/results"
    run_id: str = "run_001"
    save_every: int = 10
    save_format: str = "xdmf"
    petsc_solver: str = "cg"
    petsc_preconditioner: str = "hypre"


@app.get("/status")
async def status():
    try:
        import dolfinx
        return {
            "status": "ready",
            "dolfinx_version": dolfinx.__version__,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/run")
async def run_simulation(job: SimulationJob):
    """Execute a FEM simulation and return results."""
    try:
        cfg = HeatConfig(
            dim=job.dim, nx=job.nx, ny=job.ny, nz=job.nz,
            rho=job.rho, cp=job.cp, k=job.k, source=job.source,
            bcs=job.bcs if job.bcs else _default_bcs(job.dim),
            u_init=job.u_init,
            t_start=job.t_start, t_end=job.t_end,
            dt=job.dt, theta=job.theta,
            element_degree=job.element_degree,
            output_dir=job.output_dir,
            run_id=job.run_id,
            save_every=job.save_every,
            save_format=job.save_format,
            petsc_solver=job.petsc_solver,
            petsc_preconditioner=job.petsc_preconditioner,
        )

        solver = HeatEquationSolver(cfg)
        result = solver.solve()

        return result.to_dict()

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")


@app.get("/results/{run_id}")
async def get_result(run_id: str):
    """Retrieve result metadata for a completed run."""
    result_path = Path("/workspace/results") / run_id / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    with open(result_path) as f:
        return json.load(f)


def _default_bcs(dim: int) -> list:
    if dim == 2:
        return [
            {"type": "dirichlet", "value": 0.0, "location": "left"},
            {"type": "dirichlet", "value": 1.0, "location": "right"},
            {"type": "neumann",   "value": 0.0, "location": "top"},
            {"type": "neumann",   "value": 0.0, "location": "bottom"},
        ]
    return [
        {"type": "dirichlet", "value": 0.0, "location": "left"},
        {"type": "dirichlet", "value": 1.0, "location": "right"},
        {"type": "neumann",   "value": 0.0, "location": "top"},
        {"type": "neumann",   "value": 0.0, "location": "bottom"},
        {"type": "neumann",   "value": 0.0, "location": "front"},
        {"type": "neumann",   "value": 0.0, "location": "back"},
    ]
