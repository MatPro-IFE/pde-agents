"""
LangChain tools for the Analytics & Visualization Agent.

Tools:
  - analyze_run          : compute stats on a completed run
  - compare_runs         : cross-compare multiple runs
  - compare_study        : analyze a full parametric study
  - suggest_next_run     : propose the best next configuration based on trends
  - compute_heat_flux    : post-process heat flux from saved field
  - export_plot          : generate and save a matplotlib figure
  - get_steady_state_time: estimate when the solution reaches steady state
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.tools import tool

try:
    from database.operations import (
        get_convergence_data, get_run, get_study_comparison_data,
        list_runs, search_runs, get_suggestions_for_run,
    )
    from database.models import RunStatus
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

RESULTS_PATH = os.getenv("RESULTS_PATH", "/workspace/results")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_result(run_id: str) -> dict:
    path = Path(RESULTS_PATH) / run_id / "result.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_config(run_id: str) -> dict:
    path = Path(RESULTS_PATH) / run_id / "config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_final_field(run_id: str) -> Optional[np.ndarray]:
    path = Path(RESULTS_PATH) / run_id / "u_final.npy"
    if not path.exists():
        return None
    return np.load(path)


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def analyze_run(run_id: str) -> str:
    """
    Perform statistical analysis on a completed simulation run.

    Computes: temperature statistics, convergence behavior,
    thermal uniformity index, estimated steady-state time.

    Args:
        run_id: The simulation run identifier.

    Returns:
        JSON with detailed analysis results.
    """
    result = _load_result(run_id)
    config = _load_config(run_id)
    u = _load_final_field(run_id)

    if not result:
        return json.dumps({"error": f"No result found for run_id={run_id!r}"})

    analysis = {
        "run_id": run_id,
        "status": result.get("status"),
        "dim": config.get("dim"),
        "mesh_resolution": f"{config.get('nx')}×{config.get('ny')}",
        "physical_params": {
            "k": config.get("k"), "rho": config.get("rho"), "cp": config.get("cp"),
            "thermal_diffusivity": (
                config.get("k", 1) / (config.get("rho", 1) * config.get("cp", 1))
            ),
        },
    }

    # Temperature stats
    if u is not None:
        analysis["temperature"] = {
            "max": float(u.max()),
            "min": float(u.min()),
            "mean": float(u.mean()),
            "std": float(u.std()),
            "uniformity_index": float(1 - u.std() / (u.max() - u.min() + 1e-12)),
        }
    else:
        analysis["temperature"] = {
            "max": result.get("max_temperature"),
            "min": result.get("min_temperature"),
            "mean": result.get("mean_temperature"),
        }

    # Convergence analysis
    convergence = result.get("convergence_history", [])
    if convergence and len(convergence) > 1:
        arr = np.array(convergence)
        delta = np.abs(np.diff(arr))
        if len(delta) > 0:
            ss_idx = np.argmax(delta < 1e-4 * arr[:-1])
            analysis["convergence"] = {
                "initial_l2": float(arr[0]),
                "final_l2": float(arr[-1]),
                "max_change": float(delta.max()),
                "min_change": float(delta.min()),
                "estimated_steady_state_step": int(ss_idx),
                "estimated_steady_state_time": float(
                    config.get("t_start", 0) + ss_idx * config.get("dt", 0.01)
                ),
            }

    # Performance
    analysis["performance"] = {
        "wall_time_s": result.get("wall_time"),
        "n_dofs": result.get("n_dofs"),
        "n_timesteps": result.get("n_timesteps"),
        "dofs_per_second": (
            result.get("n_dofs", 0) * result.get("n_timesteps", 0) /
            max(result.get("wall_time", 1), 1e-9)
        ),
    }

    return json.dumps(analysis, indent=2)


@tool
def compare_runs(run_ids_json: str) -> str:
    """
    Compare multiple simulation runs side-by-side.

    Computes pairwise differences and identifies the best-performing run
    by thermal uniformity, convergence speed, and computation time.

    Args:
        run_ids_json: JSON array of run_id strings.

    Returns:
        JSON with comparison table and ranking.
    """
    try:
        run_ids = json.loads(run_ids_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    rows = []
    for rid in run_ids:
        result = _load_result(rid)
        config = _load_config(rid)
        u = _load_final_field(rid)
        if not result:
            rows.append({"run_id": rid, "error": "not found"})
            continue
        std = float(u.std()) if u is not None else None
        span = (result.get("max_temperature", 0) - result.get("min_temperature", 0))
        rows.append({
            "run_id": rid,
            "k": config.get("k"),
            "nx": config.get("nx"),
            "dt": config.get("dt"),
            "t_max": result.get("max_temperature"),
            "t_min": result.get("min_temperature"),
            "t_mean": result.get("mean_temperature"),
            "t_std": std,
            "uniformity": float(1 - std / (span + 1e-12)) if std is not None else None,
            "wall_time": result.get("wall_time"),
            "n_dofs": result.get("n_dofs"),
        })

    # Rank by uniformity (higher = more uniform temperature distribution)
    valid = [r for r in rows if "error" not in r and r.get("uniformity") is not None]
    if valid:
        ranked = sorted(valid, key=lambda r: r["uniformity"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank_uniformity"] = i + 1

    return json.dumps({
        "n_runs": len(run_ids),
        "runs": rows,
        "best_uniformity": ranked[0]["run_id"] if valid else None,
        "fastest": min(valid, key=lambda r: r.get("wall_time") or 1e9)["run_id"] if valid else None,
    }, indent=2)


@tool
def compare_study(study_id: str) -> str:
    """
    Analyze a complete parametric study and extract insights.

    Computes sensitivity of key metrics to the swept parameter,
    identifies optimal parameter value, and summarizes trends.

    Args:
        study_id: The parametric study identifier.

    Returns:
        JSON with sensitivity analysis, optimal value, and trend summary.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    data = get_study_comparison_data(study_id)
    if not data.get("param_values"):
        return json.dumps({"error": f"No data found for study {study_id!r}"})

    params = np.array(data["param_values"])
    t_max  = np.array([v or np.nan for v in data["t_max"]])
    t_mean = np.array([v or np.nan for v in data["t_mean"]])
    times  = np.array([v or np.nan for v in data["wall_times"]])

    # Sensitivity: range / mean
    def sensitivity(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return 0.0
        return float((valid.max() - valid.min()) / (abs(valid.mean()) + 1e-12))

    # Best run = highest mean temperature (maximize heat distribution)
    best_idx = int(np.nanargmax(t_mean)) if not np.all(np.isnan(t_mean)) else 0

    # Trend direction
    valid_mask = ~np.isnan(t_mean)
    trend = "inconclusive"
    if valid_mask.sum() >= 2:
        x = params[valid_mask]
        y = t_mean[valid_mask]
        slope = float(np.polyfit(x, y, 1)[0])
        trend = "increasing" if slope > 0 else "decreasing"

    return json.dumps({
        "study_id": study_id,
        "n_runs": len(params),
        "swept_parameter": data.get("swept_parameter", "unknown"),
        "parameter_range": [float(params.min()), float(params.max())],
        "sensitivity": {
            "t_max": sensitivity(t_max),
            "t_mean": sensitivity(t_mean),
            "wall_time": sensitivity(times),
        },
        "optimal": {
            "param_value": float(params[best_idx]),
            "run_id": data["run_ids"][best_idx],
            "t_mean": float(t_mean[best_idx]) if not np.isnan(t_mean[best_idx]) else None,
        },
        "t_mean_trend": trend,
        "data": {
            "param_values": params.tolist(),
            "t_max": t_max.tolist(),
            "t_mean": t_mean.tolist(),
            "wall_times": times.tolist(),
        },
    }, indent=2)


@tool
def suggest_next_run(analysis_json: str, strategy: str = "optimize_uniformity") -> str:
    """
    Suggest configuration for the next simulation run based on prior analysis.

    Strategies:
      - "optimize_uniformity": tune params for more uniform temperature
      - "refine_mesh": increase resolution for better accuracy
      - "reduce_time": find steady state faster
      - "explore": try a new region of parameter space

    Args:
        analysis_json: JSON output from analyze_run or compare_study.
        strategy:      Optimization strategy (see above).

    Returns:
        JSON with 'suggested_config' and 'rationale'.
    """
    try:
        analysis = json.loads(analysis_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    config = analysis.get("physical_params", {})
    suggested = {}
    rationale_parts = []

    if strategy == "optimize_uniformity":
        uniformity = analysis.get("temperature", {}).get("uniformity_index", 0.5)
        if uniformity < 0.8:
            k_current = config.get("k", 1.0)
            suggested["k"] = round(k_current * 2, 4)
            rationale_parts.append(
                f"Uniformity index={uniformity:.3f} < 0.8. "
                f"Doubling conductivity k from {k_current} to {suggested['k']} "
                f"should improve thermal uniformity."
            )

    elif strategy == "refine_mesh":
        nx = analysis.get("mesh_resolution", "32×32").split("×")[0]
        try:
            nx = int(nx)
        except ValueError:
            nx = 32
        suggested["nx"] = nx * 2
        suggested["ny"] = nx * 2
        suggested["dt"] = analysis.get("dt", 0.01) / 2  # reduce dt too for stability
        rationale_parts.append(
            f"Refining mesh from {nx}² to {nx*2}². "
            f"Halving dt for numerical stability."
        )

    elif strategy == "reduce_time":
        alpha = config.get("thermal_diffusivity", 1.0)
        ss_time = analysis.get("convergence", {}).get("estimated_steady_state_time", 1.0)
        suggested["t_end"] = round(ss_time * 1.2, 4)
        rationale_parts.append(
            f"Estimated steady state at t={ss_time:.4f}. "
            f"Setting t_end={suggested['t_end']} (1.2× SS time)."
        )

    elif strategy == "explore":
        k_current = config.get("k", 1.0)
        suggested["k"] = round(k_current * 5, 4)
        suggested["source"] = 1000.0
        rationale_parts.append(
            f"Exploring new regime: k={suggested['k']} (5× current), "
            f"adding body heat generation source=1000 W/m³."
        )

    return json.dumps({
        "strategy": strategy,
        "suggested_changes": suggested,
        "rationale": " ".join(rationale_parts) or "No specific changes suggested.",
    }, indent=2)


@tool
def get_steady_state_time(run_id: str, tolerance: float = 1e-4) -> str:
    """
    Estimate the time at which a simulation reaches steady state.

    Uses the convergence history: steady state is when the relative
    change in L2 norm drops below the tolerance.

    Args:
        run_id:    Simulation run identifier.
        tolerance: Relative change threshold (default 1e-4).

    Returns:
        JSON with estimated steady-state time and step index.
    """
    if DB_AVAILABLE:
        data = get_convergence_data(run_id)
    else:
        result = _load_result(run_id)
        config = _load_config(run_id)
        history = result.get("convergence_history", [])
        dt = config.get("dt", 0.01)
        data = {
            "steps": list(range(len(history))),
            "times": [i * dt for i in range(len(history))],
            "l2_norms": history,
        }

    if not data.get("l2_norms"):
        return json.dumps({"error": f"No convergence data for {run_id!r}"})

    l2 = np.array(data["l2_norms"])
    times = np.array(data["times"])

    if len(l2) < 2:
        return json.dumps({"error": "Insufficient convergence data (< 2 steps)"})

    rel_change = np.abs(np.diff(l2)) / (np.abs(l2[:-1]) + 1e-12)
    ss_indices = np.where(rel_change < tolerance)[0]

    if len(ss_indices) > 0:
        ss_step = int(ss_indices[0])
        ss_time = float(times[ss_step])
        reached = True
    else:
        ss_step = len(l2) - 1
        ss_time = float(times[-1])
        reached = False

    return json.dumps({
        "run_id": run_id,
        "steady_state_reached": reached,
        "steady_state_step": ss_step,
        "steady_state_time": ss_time,
        "final_rel_change": float(rel_change[-1]),
        "tolerance_used": tolerance,
        "note": (
            "Steady state reached within simulation time." if reached
            else "Steady state not yet reached. Consider extending t_end."
        ),
    }, indent=2)


@tool
def list_runs_for_analysis(
    status: str = "success",
    dim: int = 0,
    k_min: float = 0.0,
    k_max: float = 0.0,
    limit: int = 10,
) -> str:
    """
    Discover which simulation runs are available to analyze.

    Call this FIRST when the user asks for analysis but doesn't specify run IDs,
    e.g.:
      - "Analyze my recent runs"         → list_runs_for_analysis()
      - "Compare all 3D simulations"     → list_runs_for_analysis(dim=3)
      - "Which run performed best?"      → list_runs_for_analysis()
      - "Look at copper plate runs"      → list_runs_for_analysis()

    Args:
        status: Filter by status — "success" (default), "failed", "running", "pending".
                Use "success" to get analyzable runs.
        dim:    Filter by dimension: 2 or 3. Use 0 to return all.
        k_min:  Minimum thermal conductivity (0 = no lower bound).
        k_max:  Maximum thermal conductivity (0 = no upper bound).
        limit:  Maximum results to return (default 10).

    Returns:
        JSON with run_ids list and summary stats, ready to pass to
        analyze_run / compare_runs / suggest_next_run.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    from database.operations import get_db
    from database.models import SimulationRun, RunResult
    from sqlalchemy import select, desc

    try:
        status_filter = RunStatus(status) if status else None

        with get_db() as db:
            stmt = (
                select(SimulationRun, RunResult)
                .outerjoin(RunResult, RunResult.run_id == SimulationRun.id)
            )
            if status_filter:
                stmt = stmt.where(SimulationRun.status == status_filter)
            if dim:
                stmt = stmt.where(SimulationRun.dim == int(dim))
            if k_min:
                stmt = stmt.where(SimulationRun.k >= float(k_min))
            if k_max:
                stmt = stmt.where(SimulationRun.k <= float(k_max))

            stmt = stmt.order_by(desc(SimulationRun.created_at)).limit(int(limit))
            rows = db.execute(stmt).all()

        summaries = []
        for run, res in rows:
            summaries.append({
                "run_id":    run.run_id,
                "dim":       run.dim,
                "mesh":      f"{run.nx}×{run.ny}" + (f"×{run.nz}" if run.nz else ""),
                "k":         run.k,
                "t_end":     run.t_end,
                "wall_time": run.wall_time,
                "t_max":     res.t_max    if res else None,
                "t_min":     res.t_min    if res else None,
                "t_mean":    res.t_mean   if res else None,
                "converged": res.converged if res else None,
                "created_at": str(run.created_at)[:19],
            })

        return json.dumps({
            "available_runs": len(summaries),
            "run_ids":   [r["run_id"] for r in summaries],
            "summaries": summaries,
            "hint": (
                "Use analyze_run(run_id) for single-run deep analysis, "
                "compare_runs(run_ids_json) to compare multiple, or "
                "suggest_next_run(run_id) for next-step recommendations."
            ),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def export_summary_report(run_ids_json: str, output_path: str = "") -> str:
    """
    Generate a JSON summary report comparing multiple runs.

    Args:
        run_ids_json: JSON array of run_id strings.
        output_path:  Optional file path to save the report.

    Returns:
        JSON summary report as string.
    """
    try:
        run_ids = json.loads(run_ids_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    report = {"runs": {}}
    for rid in run_ids:
        result = _load_result(rid)
        config = _load_config(rid)
        report["runs"][rid] = {
            "config": {k: config.get(k) for k in ("dim", "nx", "ny", "k", "dt", "t_end")},
            "result": {
                k: result.get(k)
                for k in ("status", "wall_time", "n_dofs",
                           "max_temperature", "min_temperature", "mean_temperature")
            },
        }

    report["summary"] = {
        "total_runs": len(run_ids),
        "successful": sum(
            1 for r in report["runs"].values()
            if r.get("result", {}).get("status") == "success"
        ),
    }

    report_str = json.dumps(report, indent=2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report_str)

    return report_str
