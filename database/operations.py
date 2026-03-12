"""
Database CRUD operations for the PDE simulation database.

All operations are synchronous and use SQLAlchemy sessions.
Async variants are provided where needed for the FastAPI layer.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, delete, select, update, desc, func, and_, or_
from sqlalchemy.orm import Session, sessionmaker

from database.models import (
    Base, AgentMessage, AgentName, AgentRunLog, AgentSuggestion,
    ConvergenceRecord, MessageType, ParametricStudy, RunParameter,
    RunResult, RunStatus, SimulationRun, StudyRun,
)


# ─── Engine & Session ─────────────────────────────────────────────────────────

def get_database_url() -> str:
    host     = os.getenv("POSTGRES_HOST", "localhost")
    port     = os.getenv("POSTGRES_PORT", "5432")
    db       = os.getenv("POSTGRES_DB",   "pde_simulations")
    user     = os.getenv("POSTGRES_USER", "pde_user")
    password = os.getenv("POSTGRES_PASSWORD", "pde_secret")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def create_db_engine(url: Optional[str] = None, echo: bool = False):
    url = url or get_database_url()
    return create_engine(
        url, echo=echo,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        # expire_on_commit=False keeps attribute values accessible after session close
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False,
            expire_on_commit=False,
        )
    return _SessionLocal


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=get_engine())


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager that provides a database session."""
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── SimulationRun operations ─────────────────────────────────────────────────

def create_run(
    run_id: str,
    config: dict,
    description: str = "",
    tags: list[str] | None = None,
) -> SimulationRun:
    """Register a new simulation run (status=PENDING).

    If a run with the same run_id already exists it is deleted first
    so that re-running with the same ID works correctly (overwrite semantics).
    """
    with get_db() as db:
        # Remove any stale record so we can insert fresh
        existing = db.execute(
            select(SimulationRun).where(SimulationRun.run_id == run_id)
        ).scalar_one_or_none()
        if existing is not None:
            db.execute(
                delete(SimulationRun).where(SimulationRun.run_id == run_id)
            )
            db.flush()

        run = SimulationRun(
            run_id=run_id,
            pde_type=config.get("pde_type", "heat_equation"),
            dim=config.get("dim", 2),
            description=description,
            tags=tags or [],
            status=RunStatus.PENDING,
            nx=config.get("nx"),
            ny=config.get("ny"),
            nz=config.get("nz"),
            element_degree=config.get("element_degree", 1),
            t_start=config.get("t_start", 0.0),
            t_end=config.get("t_end"),
            dt=config.get("dt"),
            theta=config.get("theta", 1.0),
            k=config.get("k"),
            rho=config.get("rho"),
            cp=config.get("cp"),
            source=config.get("source", 0.0),
            config_json=config,
            output_dir=config.get("output_dir"),
        )
        db.add(run)
        db.flush()

        # Store each config value as a queryable parameter
        for key, val in config.items():
            if isinstance(val, (int, float, str, bool)):
                dtype = type(val).__name__
                db.add(RunParameter(run_id=run.id, key=key,
                                    value=str(val), dtype=dtype))
        db.refresh(run)
        db.expunge(run)
        return run


def mark_run_started(run_id: str) -> None:
    with get_db() as db:
        db.execute(
            update(SimulationRun)
            .where(SimulationRun.run_id == run_id)
            .values(status=RunStatus.RUNNING, started_at=datetime.now(timezone.utc))
        )


def mark_run_finished(
    run_id: str,
    result_data: dict,
    status: RunStatus = RunStatus.SUCCESS,
) -> None:
    """Update run with final results from a SimulationResult dict."""
    with get_db() as db:
        # Update main run record
        db.execute(
            update(SimulationRun)
            .where(SimulationRun.run_id == run_id)
            .values(
                status=status,
                finished_at=datetime.now(timezone.utc),
                n_dofs=result_data.get("n_dofs"),
                n_timesteps=result_data.get("n_timesteps"),
                wall_time=result_data.get("wall_time"),
                error_msg=result_data.get("error_message", ""),
            )
        )

        # Fetch the primary key
        run = db.execute(
            select(SimulationRun).where(SimulationRun.run_id == run_id)
        ).scalar_one_or_none()
        if run is None:
            return

        # Upsert RunResult
        existing_result = db.execute(
            select(RunResult).where(RunResult.run_id == run.id)
        ).scalar_one_or_none()

        result_vals = dict(
            t_max=result_data.get("max_temperature"),
            t_min=result_data.get("min_temperature"),
            t_mean=result_data.get("mean_temperature"),
            final_l2_norm=(
                result_data["convergence_history"][-1]
                if result_data.get("convergence_history") else None
            ),
        )
        if existing_result:
            for k, v in result_vals.items():
                setattr(existing_result, k, v)
        else:
            db.add(RunResult(run_id=run.id, **result_vals))

        # Store convergence history
        history = result_data.get("convergence_history", [])
        for i, l2 in enumerate(history):
            db.add(ConvergenceRecord(
                run_id=run.id, step=i,
                time=run.t_start + (i + 1) * (run.dt or 1.0),
                l2_norm=l2,
            ))


def mark_run_failed(run_id: str, error_message: str) -> None:
    with get_db() as db:
        db.execute(
            update(SimulationRun)
            .where(SimulationRun.run_id == run_id)
            .values(
                status=RunStatus.FAILED,
                finished_at=datetime.now(timezone.utc),
                error_msg=error_message,
            )
        )


def get_run(run_id: str) -> Optional[SimulationRun]:
    with get_db() as db:
        run = db.execute(
            select(SimulationRun).where(SimulationRun.run_id == run_id)
        ).scalar_one_or_none()
        if run is not None:
            db.expunge(run)
        return run


def list_runs(
    status: Optional[RunStatus] = None,
    dim: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[SimulationRun]:
    with get_db() as db:
        q = select(SimulationRun)
        if status:
            q = q.where(SimulationRun.status == status)
        if dim:
            q = q.where(SimulationRun.dim == dim)
        q = q.order_by(desc(SimulationRun.created_at)).limit(limit).offset(offset)
        runs = list(db.execute(q).scalars().all())
        for r in runs:
            db.expunge(r)
        return runs


def search_runs(
    k_min: Optional[float] = None,
    k_max: Optional[float] = None,
    min_t_max: Optional[float] = None,
    tags: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict]:
    """Advanced search with joins across runs and results."""
    with get_db() as db:
        q = (
            select(SimulationRun, RunResult)
            .join(RunResult, RunResult.run_id == SimulationRun.id, isouter=True)
            .where(SimulationRun.status == RunStatus.SUCCESS)
        )
        if k_min is not None:
            q = q.where(SimulationRun.k >= k_min)
        if k_max is not None:
            q = q.where(SimulationRun.k <= k_max)
        if min_t_max is not None:
            q = q.where(RunResult.t_max >= min_t_max)
        q = q.order_by(desc(SimulationRun.created_at)).limit(limit)

        rows = db.execute(q).all()
        return [{"run": r, "result": res} for r, res in rows]


# ─── Parametric Study operations ─────────────────────────────────────────────

def create_study(
    study_id: str,
    name: str,
    swept_parameter: str,
    parameter_values: list,
    base_config: dict,
    description: str = "",
) -> ParametricStudy:
    with get_db() as db:
        study = ParametricStudy(
            study_id=study_id,
            name=name,
            description=description,
            swept_parameter=swept_parameter,
            parameter_values=parameter_values,
            base_config=base_config,
        )
        db.add(study)
        db.flush()
        db.refresh(study)
        return study


def add_run_to_study(study_id: str, run_id: str, param_value: float) -> None:
    with get_db() as db:
        study = db.execute(
            select(ParametricStudy).where(ParametricStudy.study_id == study_id)
        ).scalar_one()
        run = db.execute(
            select(SimulationRun).where(SimulationRun.run_id == run_id)
        ).scalar_one()
        existing = db.execute(
            select(StudyRun).where(
                and_(StudyRun.study_id == study.id, StudyRun.run_db_id == run.id)
            )
        ).scalar_one_or_none()
        if not existing:
            db.add(StudyRun(
                study_id=study.id,
                run_db_id=run.id,
                param_value=param_value,
            ))


def get_study_results(study_id: str) -> list[dict]:
    """Fetch all runs in a study with their results for cross-comparison."""
    with get_db() as db:
        rows = db.execute(
            select(StudyRun, SimulationRun, RunResult)
            .join(SimulationRun, SimulationRun.id == StudyRun.run_db_id)
            .join(RunResult, RunResult.run_id == SimulationRun.id, isouter=True)
            .where(
                StudyRun.study_id == db.execute(
                    select(ParametricStudy.id).where(
                        ParametricStudy.study_id == study_id
                    )
                ).scalar_one()
            )
            .order_by(StudyRun.param_value)
        ).all()

        return [
            {
                "param_value": sr.param_value,
                "run_id": run.run_id,
                "status": run.status,
                "wall_time": run.wall_time,
                "t_max": res.t_max if res else None,
                "t_min": res.t_min if res else None,
                "t_mean": res.t_mean if res else None,
            }
            for sr, run, res in rows
        ]


# ─── Agent Message logging ────────────────────────────────────────────────────

def log_message(
    sender: AgentName,
    receiver: AgentName,
    msg_type: MessageType,
    content: dict,
    raw_text: str = "",
    run_id: Optional[str] = None,
    study_id: Optional[str] = None,
    processing_ms: Optional[int] = None,
) -> AgentMessage:
    with get_db() as db:
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            run_id=run_id,
            study_id=study_id,
            content=content,
            raw_text=raw_text,
            processing_ms=processing_ms,
        )
        db.add(msg)
        db.flush()
        db.refresh(msg)
        return msg


def save_suggestion(
    source_run_id: str,
    rationale: str,
    suggested_config: dict,
    priority: int = 5,
) -> AgentSuggestion:
    with get_db() as db:
        s = AgentSuggestion(
            source_run_id=source_run_id,
            rationale=rationale,
            suggested_config=suggested_config,
            priority=priority,
        )
        db.add(s)
        db.flush()
        db.refresh(s)
        return s


def get_pending_suggestions(limit: int = 10) -> list[AgentSuggestion]:
    with get_db() as db:
        return list(db.execute(
            select(AgentSuggestion)
            .where(AgentSuggestion.accepted.is_(None))
            .order_by(AgentSuggestion.priority, AgentSuggestion.created_at)
            .limit(limit)
        ).scalars().all())


# ─── Analytics helpers ────────────────────────────────────────────────────────

def get_convergence_data(run_id: str) -> dict:
    """Return convergence history as arrays for plotting."""
    with get_db() as db:
        run = db.execute(
            select(SimulationRun).where(SimulationRun.run_id == run_id)
        ).scalar_one_or_none()
        if run is None:
            return {}
        records = db.execute(
            select(ConvergenceRecord)
            .where(ConvergenceRecord.run_id == run.id)
            .order_by(ConvergenceRecord.step)
        ).scalars().all()
        return {
            "steps": [r.step for r in records],
            "times": [r.time for r in records],
            "l2_norms": [r.l2_norm for r in records],
        }


def get_study_comparison_data(study_id: str) -> dict:
    """Return study results formatted for cross-run comparison charts."""
    rows = get_study_results(study_id)
    return {
        "param_values": [r["param_value"] for r in rows],
        "run_ids":      [r["run_id"] for r in rows],
        "t_max":        [r["t_max"] for r in rows],
        "t_min":        [r["t_min"] for r in rows],
        "t_mean":       [r["t_mean"] for r in rows],
        "wall_times":   [r["wall_time"] for r in rows],
    }


# ─── Agent Run Log operations ─────────────────────────────────────────────────

def log_agent_step(
    task_id: str,
    agent_name: str,
    step_index: int,
    step_type: str,
    content: dict,
    run_id: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
) -> None:
    """Persist one step of agent reasoning to the database. Never raises."""
    try:
        with get_db() as db:
            db.add(AgentRunLog(
                task_id=task_id,
                run_id=run_id,
                agent_name=agent_name,
                step_index=step_index,
                step_type=step_type,
                content=content,
                elapsed_ms=elapsed_ms,
            ))
    except Exception:
        pass  # Logging must never break agent execution


def backfill_task_run_id(task_id: str, run_id: str) -> None:
    """Once a run_id is known, retroactively fill it in for all steps of the task."""
    try:
        with get_db() as db:
            db.execute(
                update(AgentRunLog)
                .where(AgentRunLog.task_id == task_id)
                .values(run_id=run_id)
            )
    except Exception:
        pass


def get_agent_logs(run_id: str) -> list[dict]:
    """Return the full step-by-step trace for a run (all task_ids that touch it)."""
    with get_db() as db:
        rows = db.execute(
            select(AgentRunLog)
            .where(AgentRunLog.run_id == run_id)
            .order_by(AgentRunLog.task_id, AgentRunLog.step_index)
        ).scalars().all()
        return [
            {
                "task_id":    r.task_id,
                "agent_name": r.agent_name,
                "step_index": r.step_index,
                "step_type":  r.step_type,
                "content":    r.content,
                "elapsed_ms": r.elapsed_ms,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]


def get_agent_logs_by_task(task_id: str) -> list[dict]:
    """Return all steps for a specific task invocation."""
    with get_db() as db:
        rows = db.execute(
            select(AgentRunLog)
            .where(AgentRunLog.task_id == task_id)
            .order_by(AgentRunLog.step_index)
        ).scalars().all()
        return [
            {
                "task_id":    r.task_id,
                "run_id":     r.run_id,
                "agent_name": r.agent_name,
                "step_index": r.step_index,
                "step_type":  r.step_type,
                "content":    r.content,
                "elapsed_ms": r.elapsed_ms,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]


def list_agent_tasks(limit: int = 50) -> list[dict]:
    """List distinct agent task invocations (most recent first)."""
    with get_db() as db:
        rows = db.execute(
            select(
                AgentRunLog.task_id,
                AgentRunLog.agent_name,
                AgentRunLog.run_id,
                func.min(AgentRunLog.created_at).label("started_at"),
                func.max(AgentRunLog.created_at).label("last_step_at"),
                func.count(AgentRunLog.id).label("step_count"),
            )
            .group_by(AgentRunLog.task_id, AgentRunLog.agent_name, AgentRunLog.run_id)
            .order_by(func.max(AgentRunLog.created_at).desc())
            .limit(limit)
        ).all()
        return [
            {
                "task_id":     r.task_id,
                "agent_name":  r.agent_name,
                "run_id":      r.run_id,
                "started_at":  str(r.started_at),
                "last_step_at": str(r.last_step_at),
                "step_count":  r.step_count,
            }
            for r in rows
        ]


def get_suggestions_for_run(run_id: str) -> list[dict]:
    """Return any AgentSuggestion records that reference this run."""
    with get_db() as db:
        rows = db.execute(
            select(AgentSuggestion)
            .where(AgentSuggestion.source_run_id == run_id)
            .order_by(AgentSuggestion.priority, AgentSuggestion.created_at)
        ).scalars().all()
        return [
            {
                "id":               r.id,
                "rationale":        r.rationale,
                "suggested_config": r.suggested_config,
                "priority":         r.priority,
                "accepted":         r.accepted,
                "created_at":       str(r.created_at),
            }
            for r in rows
        ]


def db_stats() -> dict:
    """Return summary statistics about the database."""
    with get_db() as db:
        total_runs = db.execute(
            select(func.count(SimulationRun.id))
        ).scalar_one()
        success_runs = db.execute(
            select(func.count(SimulationRun.id))
            .where(SimulationRun.status == RunStatus.SUCCESS)
        ).scalar_one()
        total_studies = db.execute(
            select(func.count(ParametricStudy.id))
        ).scalar_one()
        return {
            "total_runs": total_runs,
            "successful_runs": success_runs,
            "total_studies": total_studies,
        }
