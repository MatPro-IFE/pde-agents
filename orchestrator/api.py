"""
FastAPI REST interface for the multi-agent orchestrator.

Endpoints:
  POST /run          - run a task through the multi-agent system
  POST /simulate     - directly invoke Simulation Agent
  POST /analyze      - directly invoke Analytics Agent
  POST /query        - directly invoke Database Agent
  GET  /status/{id}  - get run status
  GET  /runs         - list all runs
  GET  /health       - health check
  WS   /ws/stream    - WebSocket for streaming agent output
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from orchestrator.graph import MultiAgentOrchestrator
from agents.simulation_agent import SimulationAgent
from agents.analytics_agent import AnalyticsAgent
from agents.database_agent import DatabaseAgent

# ROOT_PATH is set to e.g. "/agents" when running behind the nginx reverse proxy.
# It ensures the Swagger UI fetches the OpenAPI spec from the correct proxied URL.
_ROOT_PATH = os.getenv("ROOT_PATH", "")

app = FastAPI(
    title="PDE Agents API",
    description="Multi-agent ecosystem for PDE simulation and analysis",
    version="1.0.0",
    docs_url=None,   # disable default; we provide a custom one below
    redoc_url=None,
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=f"{_ROOT_PATH}/openapi.json",
        title="PDE Agents API",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialized agents (avoid loading LLM at import time)
_orchestrator: Optional[MultiAgentOrchestrator] = None
_sim_agent: Optional[SimulationAgent] = None
_analytics_agent: Optional[AnalyticsAgent] = None
_db_agent: Optional[DatabaseAgent] = None

# Background job registry — keyed by job_id
# Each entry: {status, result, error, task, started_at, finished_at}
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=4)

@app.on_event("startup")
async def _startup_seed_kg():
    """Initialize and seed the knowledge graph on startup (non-blocking)."""
    import asyncio
    async def _seed():
        try:
            import time
            # Brief wait to give Neo4j time to be fully ready
            await asyncio.sleep(5)
            from knowledge_graph.graph import get_kg
            kg = get_kg()
            if kg.available:
                kg.initialize()
                result = kg.seed_if_empty()
                import logging
                logging.getLogger(__name__).info(
                    "Knowledge graph seeded on startup: %s", result
                )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "KG startup seed failed (non-fatal): %s", exc
            )
    asyncio.create_task(_seed())



def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator


def get_sim_agent() -> SimulationAgent:
    global _sim_agent
    if _sim_agent is None:
        _sim_agent = SimulationAgent()
    return _sim_agent


def get_analytics_agent() -> AnalyticsAgent:
    global _analytics_agent
    if _analytics_agent is None:
        _analytics_agent = AnalyticsAgent()
    return _analytics_agent


def get_db_agent() -> DatabaseAgent:
    global _db_agent
    if _db_agent is None:
        _db_agent = DatabaseAgent()
    return _db_agent


# ─── Request/Response Models ──────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str
    max_iterations: int = 20
    context: dict = {}


class SimulateRequest(BaseModel):
    config: dict
    description: str = ""


class AnalyzeRequest(BaseModel):
    run_ids: list[str]
    goal: str = ""


class QueryRequest(BaseModel):
    query: str


class AgentResponse(BaseModel):
    request_id: str
    status: str
    result: Any
    error: Optional[str] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check service health and Ollama connectivity."""
    import httpx
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{ollama_url}/api/tags", timeout=5)
            ollama_ok = r.status_code == 200
            models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        ollama_ok = False
        models = []

    return {
        "status": "ok",
        "ollama": "ok" if ollama_ok else "unreachable",
        "available_models": models,
        "agents": ["simulation", "analytics", "database"],
    }


@app.post("/run", response_model=AgentResponse)
async def run_task(request: TaskRequest):
    """
    Run a natural language task through the full multi-agent orchestrator.

    Example task: "Run a 2D heat equation simulation on a steel plate with
    T=300K on the left and T=500K on the right, then analyze the results."
    """
    request_id = uuid.uuid4().hex[:8]
    try:
        orch = get_orchestrator()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: orch.run(request.task, max_iterations=request.max_iterations),
        )
        return AgentResponse(
            request_id=request_id,
            status="success",
            result=result,
        )
    except Exception as e:
        return AgentResponse(
            request_id=request_id,
            status="error",
            result=None,
            error=str(e),
        )


@app.post("/run/async")
async def run_task_async(request: TaskRequest):
    """
    Submit a task to the multi-agent orchestrator and return immediately.

    Returns a job_id that can be polled via GET /jobs/{job_id}.
    This avoids HTTP timeouts for long-running tasks (>60s).
    """
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "task": request.task,
        "result": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }

    def _run():
        try:
            orch = get_orchestrator()
            result = orch.run(request.task, max_iterations=request.max_iterations)
            _jobs[job_id].update(
                status="success",
                result=result,
                finished_at=time.time(),
            )
        except Exception as exc:
            _jobs[job_id].update(
                status="error",
                error=str(exc),
                finished_at=time.time(),
            )

    asyncio.get_event_loop().run_in_executor(_executor, _run)
    return {"job_id": job_id, "status": "running", "task": request.task}


@app.post("/agent/{agent_name}/async")
async def run_agent_async(agent_name: str, request: TaskRequest):
    """
    Submit a task directly to a named agent (simulation | analytics | database)
    and return a job_id immediately.
    """
    if agent_name not in ("simulation", "analytics", "database"):
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name!r}")

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "task": request.task,
        "agent": agent_name,
        "result": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }

    def _run():
        try:
            if agent_name == "simulation":
                agent = get_sim_agent()
                result = agent.run(request.task)
            elif agent_name == "analytics":
                agent = get_analytics_agent()
                result = agent.run(request.task)
            else:
                agent = get_db_agent()
                result = agent.run(request.task)
            _jobs[job_id].update(status="success", result=result, finished_at=time.time())
        except Exception as exc:
            _jobs[job_id].update(status="error", error=str(exc), finished_at=time.time())

    asyncio.get_event_loop().run_in_executor(_executor, _run)
    return {"job_id": job_id, "status": "running", "agent": agent_name, "task": request.task}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Poll the status and result of a background job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    elapsed = (
        (job["finished_at"] or time.time()) - job["started_at"]
    )
    return {
        "job_id": job_id,
        "status": job["status"],
        "task": job["task"],
        "elapsed_s": round(elapsed, 1),
        "result": job.get("result"),
        "error": job.get("error"),
    }


@app.get("/jobs")
async def list_jobs():
    """List all submitted background jobs (most recent first)."""
    return [
        {
            "job_id": jid,
            "status": j["status"],
            "task": j["task"][:80],
            "elapsed_s": round(((j["finished_at"] or time.time()) - j["started_at"]), 1),
        }
        for jid, j in sorted(
            _jobs.items(),
            key=lambda x: x[1]["started_at"],
            reverse=True,
        )
    ]


@app.post("/simulate", response_model=AgentResponse)
async def simulate(request: SimulateRequest):
    """Directly invoke the Simulation Agent."""
    request_id = uuid.uuid4().hex[:8]
    try:
        agent = get_sim_agent()
        task = request.description or f"Run this simulation: {json.dumps(request.config)}"
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.run(task),
        )
        return AgentResponse(request_id=request_id, status="success", result=result)
    except Exception as e:
        return AgentResponse(request_id=request_id, status="error", result=None, error=str(e))


@app.post("/analyze", response_model=AgentResponse)
async def analyze(request: AnalyzeRequest):
    """Directly invoke the Analytics Agent."""
    request_id = uuid.uuid4().hex[:8]
    try:
        agent = get_analytics_agent()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.compare_and_suggest(request.run_ids, request.goal),
        )
        return AgentResponse(request_id=request_id, status="success", result=result)
    except Exception as e:
        return AgentResponse(request_id=request_id, status="error", result=None, error=str(e))


@app.post("/query", response_model=AgentResponse)
async def query_db(request: QueryRequest):
    """Directly invoke the Database Agent for a natural language query."""
    request_id = uuid.uuid4().hex[:8]
    try:
        agent = get_db_agent()
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.answer_query(request.query),
        )
        return AgentResponse(request_id=request_id, status="success", result=result)
    except Exception as e:
        return AgentResponse(request_id=request_id, status="error", result=None, error=str(e))


@app.post("/agent/{agent_name}", response_model=AgentResponse)
async def run_named_agent(agent_name: str, request: TaskRequest):
    """Run a task directly on a named agent synchronously (for quick tasks)."""
    if agent_name not in ("simulation", "analytics", "database"):
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name!r}")
    request_id = uuid.uuid4().hex[:8]
    try:
        if agent_name == "simulation":
            agent = get_sim_agent()
        elif agent_name == "analytics":
            agent = get_analytics_agent()
        else:
            agent = get_db_agent()
        result = await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: agent.run(request.task)
        )
        return AgentResponse(request_id=request_id, status="success", result=result)
    except Exception as e:
        return AgentResponse(request_id=request_id, status="error", result=None, error=str(e))


@app.get("/runs")
async def list_runs_endpoint(limit: int = 20, status: str = ""):
    """List recent simulation runs from the database."""
    try:
        from database.operations import list_runs
        from database.models import RunStatus
        st = RunStatus(status) if status else None
        runs = list_runs(status=st, limit=limit)
        return [
            {
                "run_id": r.run_id,
                "status": r.status.value,
                "dim": r.dim,
                "k": r.k,
                "wall_time": r.wall_time,
                "created_at": str(r.created_at),
            }
            for r in runs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs/{run_id}")
async def get_run_endpoint(run_id: str):
    """Get full details for a simulation run."""
    try:
        from database.operations import get_run
        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        return {
            "run_id": run.run_id,
            "status": run.status.value,
            "config": run.config_json,
            "n_dofs": run.n_dofs,
            "wall_time": run.wall_time,
            "created_at": str(run.created_at),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run Explorer API ─────────────────────────────────────────────────────────

def _minio_client():
    """Return a MinIO client using env credentials."""
    from minio import Minio
    endpoint   = os.getenv("MINIO_ENDPOINT",       "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY",     "minio_admin")
    secret_key = os.getenv("MINIO_SECRET_KEY",     "minio_secret123")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


@app.get("/explorer/runs")
async def explorer_list_runs(
    search: str = "",
    status: str = "",
    dim: int = 0,
    limit: int = 100,
):
    """
    List simulation runs enriched with agent-log step counts.
    Supports free-text search on run_id/description and filtering by status/dim.
    """
    try:
        from database.operations import list_runs, get_agent_logs
        from database.models import RunStatus
        from sqlalchemy import select, func, and_
        from database.operations import get_db
        from database.models import SimulationRun, AgentRunLog, RunResult

        with get_db() as db:
            q = select(
                SimulationRun,
                func.count(AgentRunLog.id).label("log_steps"),
            ).outerjoin(
                AgentRunLog, AgentRunLog.run_id == SimulationRun.run_id
            )

            if status:
                try:
                    q = q.where(SimulationRun.status == RunStatus(status))
                except ValueError:
                    pass
            if dim:
                q = q.where(SimulationRun.dim == dim)
            if search:
                q = q.where(SimulationRun.run_id.ilike(f"%{search}%"))

            q = (q.group_by(SimulationRun.id)
                  .order_by(SimulationRun.created_at.desc())
                  .limit(limit))

            rows = db.execute(q).all()

        return [
            {
                "run_id":      r.run_id,
                "status":      r.status.value,
                "dim":         r.dim,
                "k":           r.k,
                "nx":          r.nx,
                "ny":          r.ny,
                "nz":          r.nz,
                "t_end":       r.t_end,
                "dt":          r.dt,
                "wall_time":   r.wall_time,
                "n_dofs":      r.n_dofs,
                "created_at":  str(r.created_at),
                "description": r.description,
                "log_steps":   steps,
            }
            for r, steps in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explorer/runs/{run_id}/detail")
async def explorer_run_detail(run_id: str):
    """
    Full detail for one run: metadata + config + scalar results +
    agent log summary + MinIO file listing + recommendations.
    """
    try:
        from database.operations import (
            get_run, get_agent_logs, get_suggestions_for_run,
        )
        from database.models import RunResult, SimulationRun
        from database.operations import get_db
        from sqlalchemy import select

        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

        # Scalar results
        with get_db() as db:
            res = db.execute(
                select(RunResult).where(RunResult.run_id == run.id)
            ).scalar_one_or_none()
        results = (
            {
                "t_max": res.t_max, "t_min": res.t_min, "t_mean": res.t_mean,
                "final_l2_norm": res.final_l2_norm, "converged": res.converged,
            }
            if res else {}
        )

        # Agent logs — all steps tied to this run_id
        logs = get_agent_logs(run_id)

        # Recommendations from Analytics agent
        suggestions = get_suggestions_for_run(run_id)

        # MinIO file listing
        minio_files = []
        try:
            client = _minio_client()
            bucket = os.getenv("MINIO_BUCKET_RESULTS", "simulation-results")
            prefix = f"runs/{run_id}/"
            for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
                minio_files.append({
                    "name":         obj.object_name.replace(prefix, ""),
                    "object_name":  obj.object_name,
                    "size":         obj.size,
                    "last_modified": str(obj.last_modified),
                })
        except Exception as exc:
            minio_files = [{"error": str(exc)}]

        return {
            "run_id":      run.run_id,
            "status":      run.status.value,
            "dim":         run.dim,
            "config":      run.config_json,
            "wall_time":   run.wall_time,
            "n_dofs":      run.n_dofs,
            "n_timesteps": run.n_timesteps,
            "created_at":  str(run.created_at),
            "started_at":  str(run.started_at),
            "finished_at": str(run.finished_at),
            "error_msg":   run.error_msg,
            "description": run.description,
            "results":     results,
            "agent_logs":  logs,
            "suggestions": suggestions,
            "minio_files": minio_files,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explorer/runs/{run_id}/logs")
async def explorer_run_logs(run_id: str):
    """Return the full agent log trace for a run (all steps, all agents)."""
    try:
        from database.operations import get_agent_logs
        return {"run_id": run_id, "logs": get_agent_logs(run_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explorer/runs/{run_id}/files")
async def explorer_run_files(run_id: str):
    """List all files stored in MinIO for a run."""
    try:
        client = _minio_client()
        bucket = os.getenv("MINIO_BUCKET_RESULTS", "simulation-results")
        prefix = f"runs/{run_id}/"
        files = []
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            files.append({
                "name":          obj.object_name.replace(prefix, ""),
                "object_name":   obj.object_name,
                "size":          obj.size,
                "last_modified": str(obj.last_modified),
            })
        return {"run_id": run_id, "bucket": bucket, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explorer/search")
async def explorer_search(body: dict):
    """
    Cross-search runs in PostgreSQL + MinIO.

    Body fields (all optional):
      text    - free-text search on run_id / description
      status  - filter by run status
      dim     - filter by dimension (2 or 3)
      k_min, k_max - conductivity range
      t_max_min    - minimum peak temperature
      limit        - max results (default 50)
    """
    try:
        from database.operations import get_db
        from database.models import SimulationRun, RunResult, AgentRunLog, RunStatus
        from sqlalchemy import select, func

        search   = body.get("text", "")
        status   = body.get("status", "")
        dim      = body.get("dim", 0)
        k_min    = body.get("k_min")
        k_max    = body.get("k_max")
        t_max_min = body.get("t_max_min")
        limit    = int(body.get("limit", 50))

        with get_db() as db:
            q = (
                select(SimulationRun, RunResult,
                       func.count(AgentRunLog.id).label("log_steps"))
                .outerjoin(RunResult, RunResult.run_id == SimulationRun.id)
                .outerjoin(AgentRunLog, AgentRunLog.run_id == SimulationRun.run_id)
            )
            if status:
                try:
                    q = q.where(SimulationRun.status == RunStatus(status))
                except ValueError:
                    pass
            if dim:
                q = q.where(SimulationRun.dim == int(dim))
            if search:
                q = q.where(SimulationRun.run_id.ilike(f"%{search}%"))
            if k_min is not None:
                q = q.where(SimulationRun.k >= float(k_min))
            if k_max is not None:
                q = q.where(SimulationRun.k <= float(k_max))
            if t_max_min is not None:
                q = q.where(RunResult.t_max >= float(t_max_min))

            q = (q.group_by(SimulationRun.id, RunResult.id)
                  .order_by(SimulationRun.created_at.desc())
                  .limit(limit))

            rows = db.execute(q).all()

        return [
            {
                "run_id":     run.run_id,
                "status":     run.status.value,
                "dim":        run.dim,
                "k":          run.k,
                "t_max":      res.t_max if res else None,
                "t_min":      res.t_min if res else None,
                "wall_time":  run.wall_time,
                "created_at": str(run.created_at),
                "log_steps":  steps,
            }
            for run, res, steps in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent execution.

    Send: {"task": "...", "max_iterations": 20}
    Receive: stream of {"event": "...", "data": {...}} messages
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        task = data.get("task", "")
        max_iter = data.get("max_iterations", 20)

        orch = get_orchestrator()

        await websocket.send_json({"event": "start", "data": {"task": task}})

        def stream_sync():
            return list(orch.stream(task, max_iterations=max_iter))

        events = await asyncio.get_event_loop().run_in_executor(None, stream_sync)

        for event in events:
            await websocket.send_json({"event": "update", "data": str(event)[:2000]})

        await websocket.send_json({"event": "done", "data": {"status": "complete"}})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "data": {"error": str(e)}})
        except Exception:
            pass


# ─── Knowledge Graph endpoints ────────────────────────────────────────────────

@app.get("/kg/stats")
async def kg_stats():
    """Return knowledge graph node/relationship counts."""
    try:
        from knowledge_graph.graph import get_kg
        kg = get_kg()
        return kg.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kg/seed")
async def kg_seed():
    """
    Seed the knowledge graph with static physical knowledge (materials, known issues).
    Safe to call repeatedly — uses MERGE so no duplicates are created.
    """
    try:
        from knowledge_graph.graph import get_kg
        kg = get_kg()
        if not kg.available:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        kg.initialize()
        result = kg.seed_if_empty()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/material/{name}")
async def kg_material(name: str):
    """Look up a material by name (partial match, case-insensitive)."""
    try:
        from knowledge_graph.graph import get_kg
        kg = get_kg()
        if not kg.available:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        info = kg.get_material_info(name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Material '{name}' not found")
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/run/{run_id}/similar")
async def kg_similar_runs(run_id: str, top_k: int = 5):
    """Find runs similar to the given run_id based on its configuration."""
    try:
        from knowledge_graph.graph import get_kg
        from database.operations import get_run
        kg = get_kg()
        if not kg.available:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        run = get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
        cfg = {"k": run.k, "dim": run.dim, "nx": run.nx, "ny": run.ny}
        return kg.get_similar_runs(cfg, top_k=top_k)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kg/run/{run_id}/lineage")
async def kg_run_lineage(run_id: str):
    """Return the SPAWNED_FROM ancestry chain for a run."""
    try:
        from knowledge_graph.graph import get_kg
        kg = get_kg()
        if not kg.available:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        return kg.get_run_lineage(run_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
