"""
LangChain tools for the Simulation Agent.

These tools are the "hands" of the agent:
  - run_simulation      : execute a heat equation FEM run via the fenics-runner container
  - debug_simulation    : analyze logs and suggest fixes for failed runs
  - modify_config       : patch a simulation config JSON
  - validate_config     : check config validity before running
  - list_recent_runs    : show last N simulation runs
  - get_run_status      : poll status of a running simulation
  - cancel_run          : cancel a running simulation
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from langchain_core.tools import tool

# ─── Internal imports (available when running inside the agents container) ────
try:
    from database.operations import (
        create_run, get_run, list_runs, mark_run_failed,
        mark_run_finished, mark_run_started,
    )
    from database.models import RunStatus
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

try:
    from knowledge_graph.graph import get_kg
    from knowledge_graph.rules import check_config as _check_config_rules
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

FENICS_RUNNER_URL  = os.getenv("FENICS_RUNNER_URL",  "http://fenics-runner:8080")
RESULTS_PATH       = os.getenv("RESULTS_PATH",       "/workspace/results")
MESH_PATH          = os.getenv("MESH_PATH",          "/workspace/meshes")
MINIO_ENDPOINT     = os.getenv("MINIO_ENDPOINT",     "minio:9000")
MINIO_ACCESS_KEY   = os.getenv("MINIO_ACCESS_KEY",   "minio_admin")
MINIO_SECRET_KEY   = os.getenv("MINIO_SECRET_KEY",   "minio_secret123")
MINIO_BUCKET       = os.getenv("MINIO_BUCKET_RESULTS", "simulation-results")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _call_fenics_api(endpoint: str, payload: dict, timeout: int = 3600) -> dict:
    """POST to the FEniCS runner REST API."""
    url = f"{FENICS_RUNNER_URL}/{endpoint.lstrip('/')}"
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": str(e), "detail": e.response.text}
    except httpx.RequestError as e:
        return {"error": f"Connection error: {e}"}


def _upload_run_to_minio(run_id: str, result: dict) -> dict:
    """
    Upload all output files from a completed simulation run to MinIO.

    Called automatically after every successful run.  Failures are
    non-fatal — they are logged in the returned dict but do not raise.

    Uploads to: simulation-results/runs/{run_id}/{filename}
    """
    uploaded: list[str] = []
    skipped:  list[str] = []
    errors:   list[str] = []

    if not MINIO_AVAILABLE:
        return {"minio": "skipped", "reason": "minio package not installed"}

    try:
        client = Minio(MINIO_ENDPOINT,
                       access_key=MINIO_ACCESS_KEY,
                       secret_key=MINIO_SECRET_KEY,
                       secure=False)

        # Create bucket if it doesn't exist yet
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)

        # Collect every output file the solver wrote
        output_files: list[str] = result.get("output_files", [])

        # Also include the standard files that are always written but may not
        # appear in output_files for older runs
        run_dir = Path(RESULTS_PATH) / run_id
        always_include = [
            run_dir / "config.json",
            run_dir / "result.json",
            run_dir / "u_final.npy",
            run_dir / "dof_coords.npy",
            run_dir / "snapshot_times.npy",
        ]
        all_paths = {Path(f) for f in output_files} | set(always_include)

        # Upload snapshots directory
        snapshots_dir = run_dir / "snapshots"
        if snapshots_dir.exists():
            all_paths.update(snapshots_dir.glob("u_*.npy"))

        prefix = f"runs/{run_id}"
        for path in sorted(all_paths):
            if not path.exists():
                skipped.append(path.name)
                continue
            # Preserve sub-directory structure (e.g. snapshots/u_0000.npy)
            try:
                rel = path.relative_to(run_dir)
            except ValueError:
                rel = Path(path.name)
            obj_name = f"{prefix}/{rel}"
            try:
                client.fput_object(MINIO_BUCKET, obj_name, str(path))
                uploaded.append(obj_name)
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

    except Exception as exc:
        return {"minio": "error", "reason": str(exc)}

    return {
        "minio": "ok",
        "bucket": MINIO_BUCKET,
        "prefix": f"runs/{run_id}",
        "uploaded": len(uploaded),
        "skipped": len(skipped),
        "errors": errors,
    }


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def run_simulation(config_json: str) -> str:
    """
    Launch a FEM heat equation simulation.

    Args:
        config_json: JSON string with simulation configuration.
            Required fields: dim (2 or 3), nx, ny, t_end, dt, k
            Optional: nz (3D), rho, cp, source, bcs, theta, element_degree,
                      run_id, output_dir, save_format

    Returns:
        JSON string with run_id, status, and result summary.
    """
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    # Assign run_id if not provided
    if "run_id" not in config:
        config["run_id"] = _new_run_id("heat")
    config.setdefault("output_dir", RESULTS_PATH)

    run_id = config["run_id"]

    # Register in DB
    if DB_AVAILABLE:
        create_run(run_id, config)
        mark_run_started(run_id)

    # Call FEniCS runner
    result = _call_fenics_api("run", config)

    if "error" in result:
        if DB_AVAILABLE:
            mark_run_failed(run_id, result["error"])
        return json.dumps({
            "run_id": run_id,
            "status": "failed",
            "error": result["error"],
        })

    if DB_AVAILABLE:
        mark_run_finished(run_id, result, status=RunStatus.SUCCESS)

    # Always upload output files to MinIO so every run is archived,
    # regardless of whether the Database Agent is involved.
    minio_info = _upload_run_to_minio(run_id, result)

    # Auto-populate the knowledge graph after every successful run.
    kg_info: dict = {"kg": "skipped"}
    if KG_AVAILABLE:
        try:
            kg = get_kg()
            if kg.available:
                # Attach any pre-run rule warnings so the graph knows which
                # issues were present in this configuration.
                warnings = _check_config_rules(config)
                added = kg.add_run(run_id, config, result, warnings=warnings)
                kg_info = {"kg": "ok" if added else "error"}
        except Exception as exc:
            kg_info = {"kg": "error", "reason": str(exc)}

    return json.dumps({
        "run_id": run_id,
        "status": "success",
        "summary": result.get("summary", ""),
        "wall_time": result.get("wall_time"),
        "n_dofs": result.get("n_dofs"),
        "t_max": result.get("max_temperature"),
        "t_min": result.get("min_temperature"),
        "output_dir": config["output_dir"],
        "minio": minio_info,
        **kg_info,
    })


@tool
def validate_config(config_json: str) -> str:
    """
    Validate a simulation configuration without running it.

    Checks for required fields, physical plausibility, and resource estimates.

    Args:
        config_json: JSON string with simulation configuration.

    Returns:
        JSON string with 'valid' (bool), 'errors' (list), 'warnings' (list),
        and 'estimates' (DOF count, memory, estimated wall time).
    """
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        return json.dumps({"valid": False, "errors": [f"JSON parse error: {e}"]})

    errors, warnings = [], []

    # Required fields
    for field in ("dim", "nx", "ny", "t_end", "dt"):
        if field not in config:
            errors.append(f"Missing required field: '{field}'")

    dim = config.get("dim", 2)
    if dim not in (2, 3):
        errors.append(f"'dim' must be 2 or 3, got {dim}")

    nx = config.get("nx", 32)
    ny = config.get("ny", 32)
    nz = config.get("nz", 1) if dim == 3 else 1
    degree = config.get("element_degree", 1)

    # DOF estimate
    if dim == 2:
        n_dofs_est = (nx + 1) * (ny + 1) if degree == 1 else (2*nx + 1) * (2*ny + 1)
    else:
        n_dofs_est = (nx+1)*(ny+1)*(nz+1)

    dt   = config.get("dt", 0.01)
    t_end = config.get("t_end", 1.0)
    n_steps = int(t_end / dt) if dt > 0 else 0

    # Physical plausibility
    if config.get("k", 1.0) <= 0:
        errors.append("Thermal conductivity k must be > 0")
    if config.get("rho", 1.0) <= 0:
        errors.append("Density rho must be > 0")
    if config.get("cp", 1.0) <= 0:
        errors.append("Specific heat cp must be > 0")
    if dt <= 0:
        errors.append("dt must be > 0")
    if t_end <= 0:
        errors.append("t_end must be > 0")

    # CFL-like stability warning for explicit methods
    theta = config.get("theta", 1.0)
    if theta < 0.5:
        warnings.append(
            f"θ={theta} < 0.5: explicit scheme may be unstable. "
            f"Use θ≥0.5 (Crank-Nicolson or Backward Euler)."
        )

    # Resource estimates
    mem_mb_est = n_dofs_est * 8 * 3 / 1e6  # ~3 vectors of float64
    time_est_s = n_steps * n_dofs_est * 1e-7  # very rough estimate

    if n_dofs_est > 10_000_000:
        warnings.append(f"Large problem: ~{n_dofs_est:,} DOFs. Consider reducing mesh resolution.")

    return json.dumps({
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "estimates": {
            "n_dofs": n_dofs_est,
            "n_timesteps": n_steps,
            "memory_mb": round(mem_mb_est, 1),
            "wall_time_estimate_s": round(time_est_s, 1),
        },
    })


@tool
def modify_config(config_json: str, changes_json: str) -> str:
    """
    Apply a patch of changes to a simulation configuration.

    Args:
        config_json:  Original configuration as JSON string.
        changes_json: Dict of fields to update, as JSON string.
                      Example: '{"k": 2.5, "nx": 64, "run_id": "run_v2"}'

    Returns:
        Updated configuration as JSON string.
    """
    try:
        config  = json.loads(config_json)
        changes = json.loads(changes_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON parse error: {e}"})

    config.update(changes)
    if "run_id" not in changes:
        config["run_id"] = _new_run_id("modified")

    return json.dumps(config, indent=2)


@tool
def debug_simulation(run_id: str) -> str:
    """
    Analyze a failed or slow simulation run and suggest fixes.

    Reads the run log and result files, then produces a structured
    diagnosis with suggested configuration changes.

    Args:
        run_id: The run identifier to debug.

    Returns:
        JSON with 'diagnosis' (str), 'likely_cause' (str),
        and 'suggested_fixes' (list of config patches).
    """
    result_dir = Path(RESULTS_PATH) / run_id
    log_path   = result_dir / "solver.log"
    cfg_path   = result_dir / "config.json"
    res_path   = result_dir / "result.json"

    diagnosis = []
    likely_cause = "unknown"
    suggested_fixes = []

    # Load config
    config = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)
    else:
        return json.dumps({
            "diagnosis": f"No config found at {cfg_path}",
            "likely_cause": "run not found",
            "suggested_fixes": [],
        })

    # Load result if available
    result = {}
    if res_path.exists():
        with open(res_path) as f:
            result = json.load(f)

    # Read log if available
    log_text = ""
    if log_path.exists():
        with open(log_path) as f:
            log_text = f.read()[-5000:]  # last 5k chars

    # Heuristic diagnosis
    if result.get("status") == "failed":
        error = result.get("error_message", "")
        if "diverged" in error.lower() or "nan" in log_text.lower():
            likely_cause = "numerical_divergence"
            diagnosis.append("Solution diverged (NaN detected).")
            dt = config.get("dt", 0.01)
            k  = config.get("k", 1.0)
            rho_cp = config.get("rho", 1.0) * config.get("cp", 1.0)
            suggested_fixes.append({
                "description": "Reduce time step by 5x",
                "changes": {"dt": round(dt / 5, 6)},
            })
            if config.get("theta", 1.0) < 1.0:
                suggested_fixes.append({
                    "description": "Switch to fully implicit (Backward Euler)",
                    "changes": {"theta": 1.0},
                })

        elif "memory" in error.lower() or "killed" in log_text.lower():
            likely_cause = "out_of_memory"
            diagnosis.append("Process killed, likely OOM.")
            nx = config.get("nx", 32)
            suggested_fixes.append({
                "description": "Halve mesh resolution",
                "changes": {"nx": nx // 2, "ny": config.get("ny", nx) // 2},
            })

        elif "convergence" in error.lower():
            likely_cause = "solver_convergence"
            diagnosis.append("Linear solver failed to converge.")
            suggested_fixes.append({
                "description": "Switch to direct solver (MUMPS)",
                "changes": {"petsc_solver": "preonly",
                            "petsc_preconditioner": "lu"},
            })
    else:
        # Run succeeded but may be slow or produce unexpected results
        convergence = result.get("convergence_history", [])
        if convergence and len(convergence) > 10:
            last = convergence[-1]
            first = convergence[0]
            if last > first * 0.99:
                likely_cause = "slow_convergence"
                diagnosis.append(
                    f"L2 norm barely changed: {first:.4g} → {last:.4g}. "
                    "Steady state may not be reached."
                )
                suggested_fixes.append({
                    "description": "Extend simulation time",
                    "changes": {"t_end": config.get("t_end", 1.0) * 5},
                })

    if not diagnosis:
        diagnosis.append("No obvious issue detected. Check output files manually.")

    return json.dumps({
        "run_id": run_id,
        "diagnosis": " ".join(diagnosis),
        "likely_cause": likely_cause,
        "config_snapshot": {
            "k": config.get("k"), "dt": config.get("dt"),
            "nx": config.get("nx"), "theta": config.get("theta"),
        },
        "suggested_fixes": suggested_fixes,
    }, indent=2)


@tool
def list_recent_runs(limit: int = 10, status_filter: str = "") -> str:
    """
    List recent simulation runs.

    Args:
        limit:         Number of runs to return (default 10).
        status_filter: Filter by status: "pending", "running", "success", "failed", or "" for all.

    Returns:
        JSON list of run summaries.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    status = None
    if status_filter:
        try:
            status = RunStatus(status_filter)
        except ValueError:
            return json.dumps({"error": f"Unknown status '{status_filter}'"})

    runs = list_runs(status=status, limit=limit)
    return json.dumps([
        {
            "run_id": r.run_id,
            "status": r.status.value,
            "dim": r.dim,
            "k": r.k,
            "nx": r.nx,
            "t_end": r.t_end,
            "wall_time": r.wall_time,
            "created_at": str(r.created_at),
        }
        for r in runs
    ], indent=2)


@tool
def get_run_status(run_id: str) -> str:
    """
    Get the current status and summary of a simulation run.

    Args:
        run_id: The simulation run identifier.

    Returns:
        JSON with run status, config snapshot, and result metrics.
    """
    if not DB_AVAILABLE:
        # Fall back to filesystem
        result_dir = Path(RESULTS_PATH) / run_id
        res_path = result_dir / "result.json"
        if res_path.exists():
            with open(res_path) as f:
                return f.read()
        return json.dumps({"error": f"Run {run_id!r} not found"})

    run = get_run(run_id)
    if run is None:
        return json.dumps({"error": f"Run {run_id!r} not found in database"})

    return json.dumps({
        "run_id": run.run_id,
        "status": run.status.value,
        "dim": run.dim,
        "n_dofs": run.n_dofs,
        "wall_time": run.wall_time,
        "t_end": run.t_end,
        "k": run.k,
        "created_at": str(run.created_at),
        "output_dir": run.output_dir,
    })


@tool
def run_parametric_sweep(
    swept_parameter: str,
    values_json: str,
    base_config_json: str,
) -> str:
    """
    Launch a parametric sweep: run the simulation for each value of a parameter.

    Args:
        swept_parameter: Config key to vary (e.g. "k", "dt", "nx").
        values_json:     JSON array of values to sweep over.
        base_config_json: Base config as JSON string (common settings).

    Returns:
        JSON summary with study_id and list of run_ids.
    """
    try:
        values      = json.loads(values_json)
        base_config = json.loads(base_config_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON parse error: {e}"})

    study_id = _new_run_id("study")
    run_ids  = []

    for val in values:
        cfg = dict(base_config)
        cfg[swept_parameter] = val
        cfg["run_id"] = f"{study_id}_{swept_parameter}_{str(val).replace('.', 'p')}"

        # Use run_simulation tool logic directly
        result_raw = run_simulation.invoke(json.dumps(cfg))
        result = json.loads(result_raw)
        run_ids.append({
            "value": val,
            "run_id": result.get("run_id"),
            "status": result.get("status"),
        })

    return json.dumps({
        "study_id": study_id,
        "swept_parameter": swept_parameter,
        "values": values,
        "runs": run_ids,
    }, indent=2)
