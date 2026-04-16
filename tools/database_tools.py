"""
LangChain tools for the Database Agent.

Tools:
  - store_result         : save simulation result to database and MinIO
  - query_runs           : SQL-like query over simulation runs
  - catalog_study        : register and organize a parametric study
  - fetch_run_data       : retrieve full data for a run
  - export_to_csv        : export query results to CSV
  - delete_run           : remove a run (soft-delete)
  - db_health_check      : verify database connectivity and stats
  - upload_to_minio      : upload large files to object storage
"""

from __future__ import annotations

import csv
import json
import os


def _safe_json_parse(value, fallback=None):
    """Parse a JSON string robustly — handles str, dict, list, double-encoded inputs."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return fallback if fallback is not None else {}
    try:
        parsed = json.loads(value)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        return fallback
from io import StringIO
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

try:
    from database.operations import (
        create_run, create_study, db_stats, get_convergence_data,
        get_run, get_study_results, init_db, list_runs, log_message,
        mark_run_failed, mark_run_finished, save_suggestion,
        search_runs, add_run_to_study,
        get_agent_logs, list_agent_tasks, get_suggestions_for_run,
    )
    from database.models import AgentName, MessageType, RunStatus
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

RESULTS_PATH = os.getenv("RESULTS_PATH", "/workspace/results")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio_admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio_secret123")
MINIO_BUCKET_RESULTS = os.getenv("MINIO_BUCKET_RESULTS", "simulation-results")
MINIO_BUCKET_MESHES  = os.getenv("MINIO_BUCKET_MESHES", "meshes")


def _get_minio_client():
    if not MINIO_AVAILABLE:
        raise RuntimeError("minio package not installed")
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def store_result(result_json: str) -> str:
    """
    Persist a simulation result to the PostgreSQL database and upload
    output files to MinIO object storage.

    Args:
        result_json: JSON string of a SimulationResult dict.

    Returns:
        JSON with 'stored' (bool), 'db_id', 'minio_prefix'.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    result = _safe_json_parse(result_json)
    if result is None:
        return json.dumps({"error": "JSON parse error in result_json"})

    run_id = result.get("run_id")
    if not run_id:
        return json.dumps({"error": "result_json missing 'run_id'"})

    try:
        status = RunStatus.SUCCESS if result.get("status") == "success" else RunStatus.FAILED
        mark_run_finished(run_id, result, status=status)

        # Upload output files to MinIO
        minio_prefix = f"runs/{run_id}"
        uploaded = []
        if MINIO_AVAILABLE:
            client = _get_minio_client()
            # Ensure bucket exists
            if not client.bucket_exists(MINIO_BUCKET_RESULTS):
                client.make_bucket(MINIO_BUCKET_RESULTS)
            for fpath in result.get("output_files", []):
                p = Path(fpath)
                if p.exists():
                    obj_name = f"{minio_prefix}/{p.name}"
                    client.fput_object(MINIO_BUCKET_RESULTS, obj_name, str(p))
                    uploaded.append(obj_name)

        return json.dumps({
            "stored": True,
            "run_id": run_id,
            "minio_prefix": minio_prefix,
            "uploaded_files": uploaded,
        })
    except Exception as e:
        return json.dumps({"error": str(e), "stored": False})


@tool
def query_runs(
    status: str = "",
    dim: int = 0,
    limit: int = 20,
) -> str:
    """
    List recent simulation runs with basic metadata.

    For richer filtering (by k, T_max, text search), use search_history instead.

    Args:
        status: "success", "failed", "running", or "pending". Empty = all.
        dim:    2 or 3. Use 0 for all.
        limit:  Max rows to return (default 20).

    Returns:
        JSON array of runs with run_id, status, dim, k, t_end, wall_time, created_at.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    status_filter = RunStatus(status) if status else None
    runs = list_runs(
        status=status_filter,
        dim=dim if dim else None,
        limit=limit,
    )

    return json.dumps([
        {
            "run_id":    r.run_id,
            "status":    r.status.value,
            "dim":       r.dim,
            "k":         r.k,
            "nx":        r.nx,
            "t_end":     r.t_end,
            "wall_time": r.wall_time,
            "n_dofs":    r.n_dofs,
            "created_at": str(r.created_at),
        }
        for r in runs
    ], indent=2)


@tool
def catalog_study(
    study_id: str,
    name: str,
    swept_parameter: str,
    values_json: str,
    base_config_json: str,
    run_ids_json: str,
    description: str = "",
) -> str:
    """
    Register a parametric study in the database and link its runs.

    Args:
        study_id:          Unique identifier for the study.
        name:              Human-readable name.
        swept_parameter:   The parameter that was varied.
        values_json:       JSON array of the swept values.
        base_config_json:  Base config dict as JSON string.
        run_ids_json:      JSON array of [run_id, param_value] pairs.
        description:       Optional study description.

    Returns:
        JSON with study registration confirmation.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    values = _safe_json_parse(values_json, fallback=[])
    base_config = _safe_json_parse(base_config_json)
    run_pairs = _safe_json_parse(run_ids_json, fallback=[])
    if not values or base_config is None:
        return json.dumps({"error": "JSON parse error in values or base_config"})

    try:
        study = create_study(
            study_id=study_id,
            name=name,
            swept_parameter=swept_parameter,
            parameter_values=values,
            base_config=base_config,
            description=description,
        )

        for run_id, param_val in run_pairs:
            add_run_to_study(study_id, run_id, param_val)

        return json.dumps({
            "cataloged": True,
            "study_id": study_id,
            "n_runs": len(run_pairs),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "cataloged": False})


@tool
def fetch_run_data(run_id: str, include_convergence: bool = True) -> str:
    """
    Retrieve the complete record for a simulation run.

    Args:
        run_id:              Simulation run identifier.
        include_convergence: Whether to include timestep convergence history.

    Returns:
        JSON with full run record including config, results, and optionally
        the convergence history array.
    """
    if not DB_AVAILABLE:
        # Fall back to filesystem
        result_path = Path(RESULTS_PATH) / run_id / "result.json"
        config_path = Path(RESULTS_PATH) / run_id / "config.json"
        out = {}
        if result_path.exists():
            with open(result_path) as f:
                out["result"] = json.load(f)
        if config_path.exists():
            with open(config_path) as f:
                out["config"] = json.load(f)
        return json.dumps(out, indent=2)

    run = get_run(run_id)
    if run is None:
        return json.dumps({"error": f"Run {run_id!r} not found"})

    data = {
        "run_id": run.run_id,
        "status": run.status.value,
        "created_at": str(run.created_at),
        "dim": run.dim,
        "n_dofs": run.n_dofs,
        "wall_time": run.wall_time,
        "config": run.config_json,
    }

    if run.results:
        data["results"] = {
            "t_max": run.results.t_max,
            "t_min": run.results.t_min,
            "t_mean": run.results.t_mean,
            "final_l2_norm": run.results.final_l2_norm,
        }

    if include_convergence:
        data["convergence"] = get_convergence_data(run_id)

    return json.dumps(data, indent=2)


@tool
def export_to_csv(
    status: str = "",
    dim: int = 0,
    limit: int = 1000,
    output_path: str = "",
) -> str:
    """
    Export simulation run data to CSV format.

    Args:
        status:      Filter by status ("success", "failed", etc.). Empty = all.
        dim:         Filter by dimension (2, 3, or 0 for all).
        limit:       Maximum rows to export (default 1000).
        output_path: Optional file path to save CSV. If empty, returns CSV text.

    Returns:
        CSV string or confirmation of file save.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    runs = list_runs(
        status=RunStatus(status) if status else None,
        dim=dim if dim else None,
        limit=limit,
    )

    fieldnames = ["run_id", "status", "dim", "k", "nx", "ny", "dt",
                  "t_end", "wall_time", "n_dofs", "created_at"]

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in runs:
        writer.writerow({
            "run_id": r.run_id,
            "status": r.status.value,
            "dim": r.dim,
            "k": r.k,
            "nx": r.nx,
            "ny": r.ny,
            "dt": r.dt,
            "t_end": r.t_end,
            "wall_time": r.wall_time,
            "n_dofs": r.n_dofs,
            "created_at": str(r.created_at),
        })

    csv_str = buf.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(csv_str)
        return json.dumps({"saved": True, "path": output_path, "n_rows": len(runs)})

    return csv_str


@tool
def db_health_check() -> str:
    """
    Check database connectivity and return summary statistics.

    Returns:
        JSON with 'healthy' (bool), connection info, and table counts.
    """
    if not DB_AVAILABLE:
        return json.dumps({"healthy": False, "error": "Database module not available"})

    try:
        stats = db_stats()
        return json.dumps({
            "healthy": True,
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "database": os.getenv("POSTGRES_DB", "pde_simulations"),
            **stats,
        })
    except Exception as e:
        return json.dumps({"healthy": False, "error": str(e)})


@tool
def search_history(
    status: str = "",
    dim: int = 0,
    text: str = "",
    k_min: float = 0.0,
    k_max: float = 0.0,
    t_max_min: float = 0.0,
    limit: int = 20,
) -> str:
    """
    Search the simulation run history database.

    Use this whenever the user asks questions like:
      - "What runs have I done?" → call with no arguments
      - "Show me all failed runs" → status="failed"
      - "List all 3D runs" → dim=3
      - "Find runs where k > 50" → k_min=50
      - "Which run had T > 1000K?" → t_max_min=1000
      - "Show runs with 'steel' in the name" → text="steel"

    Args:
        status:    Filter by status: "success", "failed", "running", "pending".
                   Leave empty to return all statuses.
        dim:       Filter by dimension: 2 or 3. Use 0 to return all.
        text:      Substring to match anywhere in the run_id (case-insensitive).
        k_min:     Minimum thermal conductivity (0 = no lower bound).
        k_max:     Maximum thermal conductivity (0 = no upper bound).
        t_max_min: Only return runs where the recorded peak temperature
                   is at least this value (0 = no filter).
        limit:     Maximum number of rows to return (default 20).

    Returns:
        JSON with total_found and a list of runs. Each run contains:
        run_id, status, dim, mesh, k, t_max, t_min, t_mean, wall_time,
        n_dofs, converged, error_msg, created_at.
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
            if text:
                stmt = stmt.where(SimulationRun.run_id.ilike(f"%{text}%"))
            if k_min:
                stmt = stmt.where(SimulationRun.k >= float(k_min))
            if k_max:
                stmt = stmt.where(SimulationRun.k <= float(k_max))
            if t_max_min:
                stmt = stmt.where(RunResult.t_max >= float(t_max_min))

            stmt = stmt.order_by(desc(SimulationRun.created_at)).limit(int(limit))
            rows = db.execute(stmt).all()

        results = []
        for run, res in rows:
            results.append({
                "run_id":    run.run_id,
                "status":    run.status.value,
                "dim":       run.dim,
                "mesh":      f"{run.nx}×{run.ny}" + (f"×{run.nz}" if run.nz else ""),
                "k":         run.k,
                "rho":       run.rho,
                "cp":        run.cp,
                "t_end":     run.t_end,
                "dt":        run.dt,
                "n_dofs":    run.n_dofs,
                "wall_time": run.wall_time,
                "t_max":     res.t_max          if res else None,
                "t_min":     res.t_min          if res else None,
                "t_mean":    res.t_mean         if res else None,
                "l2_norm":   res.final_l2_norm  if res else None,
                "converged": res.converged      if res else None,
                "error_msg": run.error_msg or "",
                "created_at": str(run.created_at)[:19],
            })

        return json.dumps({"total_found": len(results), "runs": results}, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_run_summary(run_id: str) -> str:
    """
    Get a comprehensive human-readable summary of one simulation run.

    Returns configuration, results, agent decision log summary, and
    any Analytics agent recommendations — everything needed to understand
    what the run was, how it went, and what was suggested next.

    Use this when the user asks:
      - "Tell me about run <run_id>"
      - "What happened in <run_id>?"
      - "What did the agents do for <run_id>?"
      - "What were the recommendations for <run_id>?"

    Args:
        run_id: The simulation run identifier.

    Returns:
        JSON with: config, results, agent_log_summary, suggestions.
    """
    if not DB_AVAILABLE:
        return json.dumps({"error": "Database not available"})

    try:
        run = get_run(run_id)
        if not run:
            return json.dumps({"error": f"Run '{run_id}' not found in database"})

        # Scalar results
        from database.operations import get_db
        from database.models import RunResult
        from sqlalchemy import select
        with get_db() as db:
            res = db.execute(
                select(RunResult).where(RunResult.run_id == run.id)
            ).scalar_one_or_none()

        results = {}
        if res:
            results = {
                "t_max": res.t_max, "t_min": res.t_min, "t_mean": res.t_mean,
                "final_l2_norm": res.final_l2_norm, "converged": res.converged,
            }

        # Agent log summary
        logs = get_agent_logs(run_id)
        step_types = {}
        agents_involved = set()
        for log in logs:
            step_types[log["step_type"]] = step_types.get(log["step_type"], 0) + 1
            agents_involved.add(log["agent_name"])

        # Final answer from logs (if any)
        final_answers = [
            l["content"].get("answer", "")
            for l in logs if l["step_type"] == "final_answer"
        ]

        # Recommendations
        suggestions = get_suggestions_for_run(run_id)

        return json.dumps({
            "run_id":          run.run_id,
            "status":          run.status.value,
            "created_at":      str(run.created_at)[:19],
            "wall_time_s":     run.wall_time,
            "config": {
                "pde_type":  run.config_json.get("pde_type", "heat_equation"),
                "dim":       run.dim,
                "mesh":      f"{run.nx}×{run.ny}" + (f"×{run.nz}" if run.nz else ""),
                "k":         run.k,
                "rho":       run.rho,
                "cp":        run.cp,
                "t_end":     run.t_end,
                "dt":        run.dt,
                "theta":     run.theta,
                "bcs":       run.config_json.get("bcs", []),
                "source":    run.source,
            },
            "results":         results,
            "error":           run.error_msg or "",
            "agent_activity": {
                "total_steps":      len(logs),
                "agents_involved":  list(agents_involved),
                "step_breakdown":   step_types,
                "final_answers":    final_answers,
            },
            "recommendations": suggestions,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def upload_to_minio(local_path: str, bucket: str = "", object_name: str = "") -> str:
    """
    Upload a file to MinIO object storage.

    Args:
        local_path:  Absolute path to the local file.
        bucket:      MinIO bucket name (default: simulation-results).
        object_name: Object key in the bucket (default: file basename).

    Returns:
        JSON with upload confirmation and object URL.
    """
    if not MINIO_AVAILABLE:
        return json.dumps({"error": "minio package not installed"})

    bucket = bucket or MINIO_BUCKET_RESULTS
    p = Path(local_path)
    if not p.exists():
        return json.dumps({"error": f"File not found: {local_path}"})

    object_name = object_name or p.name

    try:
        client = _get_minio_client()
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        client.fput_object(bucket, object_name, local_path)
        return json.dumps({
            "uploaded": True,
            "bucket": bucket,
            "object": object_name,
            "url": f"http://{MINIO_ENDPOINT}/{bucket}/{object_name}",
        })
    except Exception as e:
        return json.dumps({"uploaded": False, "error": str(e)})
