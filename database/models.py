"""
SQLAlchemy ORM models for the PDE simulation database.

Tables:
  - simulation_runs     : metadata for each simulation run
  - run_parameters      : key-value config store per run
  - run_results         : scalar results (T_max, T_min, wall_time, etc.)
  - convergence_history : per-timestep L2 norm log
  - parametric_studies  : groups of related runs for sweep analysis
  - study_runs          : many-to-many: study ↔ run
  - agent_messages      : log of inter-agent communications
  - agent_suggestions   : suggestions emitted by Analytics agent
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Enum, Float, ForeignKey,
    Index, Integer, JSON, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ─── Enums ────────────────────────────────────────────────────────────────────

class RunStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCESS   = "success"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class AgentName(str, enum.Enum):
    SIMULATION = "simulation"
    ANALYTICS  = "analytics"
    DATABASE   = "database"
    ORCHESTRATOR = "orchestrator"


class MessageType(str, enum.Enum):
    TASK      = "task"
    RESULT    = "result"
    ERROR     = "error"
    SUGGESTION = "suggestion"
    QUERY     = "query"


# ─── Core Tables ──────────────────────────────────────────────────────────────

class SimulationRun(Base):
    """Top-level record for a single simulation run."""
    __tablename__ = "simulation_runs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String(128), nullable=False, unique=True, index=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at  = Column(DateTime(timezone=True), server_default=func.now(),
                         onupdate=func.now(), nullable=False)
    started_at  = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))

    # Classification
    pde_type    = Column(String(64),  default="heat_equation", nullable=False)
    dim         = Column(Integer,     nullable=False)
    description = Column(Text,        default="")
    tags        = Column(JSON,        default=list)

    # Status
    status      = Column(Enum(RunStatus), default=RunStatus.PENDING, nullable=False, index=True)
    error_msg   = Column(Text,        default="")

    # Mesh info
    nx          = Column(Integer)
    ny          = Column(Integer)
    nz          = Column(Integer)
    n_dofs      = Column(BigInteger)
    element_degree = Column(Integer, default=1)

    # Time integration
    t_start     = Column(Float, default=0.0)
    t_end       = Column(Float)
    dt          = Column(Float)
    theta       = Column(Float, default=1.0)
    n_timesteps = Column(Integer)

    # Physical params (snapshot for quick queries)
    k           = Column(Float, doc="thermal conductivity")
    rho         = Column(Float, doc="density")
    cp          = Column(Float, doc="specific heat")
    source      = Column(Float, default=0.0, doc="body heat generation")

    # Full config
    config_json = Column(JSON, default=dict)

    # Perf
    wall_time   = Column(Float, doc="seconds")

    # Experiment tracking
    experiment_phase = Column(String(64), nullable=True, index=True,
                              doc="e.g. 'ablation_v2', 'kg_growth_clean', 'production'")

    # Storage
    output_dir  = Column(String(512))
    minio_prefix = Column(String(512))

    # Relationships
    parameters         = relationship("RunParameter",       back_populates="run",
                                       cascade="all, delete-orphan")
    results            = relationship("RunResult",          back_populates="run",
                                       cascade="all, delete-orphan", uselist=False)
    convergence_history = relationship("ConvergenceRecord", back_populates="run",
                                        cascade="all, delete-orphan",
                                        order_by="ConvergenceRecord.step")
    study_runs         = relationship("StudyRun",           back_populates="run")

    def __repr__(self):
        return f"<SimulationRun id={self.id} run_id={self.run_id!r} status={self.status}>"


class RunParameter(Base):
    """Key-value store for run configuration (enables arbitrary parameter queries)."""
    __tablename__ = "run_parameters"

    id      = Column(Integer, primary_key=True)
    run_id  = Column(Integer, ForeignKey("simulation_runs.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    key     = Column(String(128), nullable=False)
    value   = Column(Text)
    dtype   = Column(String(16), default="str")  # str, float, int, bool, json

    run = relationship("SimulationRun", back_populates="parameters")

    __table_args__ = (
        UniqueConstraint("run_id", "key", name="uq_run_parameter"),
        Index("ix_run_parameters_key", "key"),
    )


class RunResult(Base):
    """Scalar results extracted from a simulation run."""
    __tablename__ = "run_results"

    id          = Column(Integer, primary_key=True)
    run_id      = Column(Integer, ForeignKey("simulation_runs.id", ondelete="CASCADE"),
                         nullable=False, unique=True)

    # Thermal field statistics
    t_max       = Column(Float, doc="maximum temperature")
    t_min       = Column(Float, doc="minimum temperature")
    t_mean      = Column(Float, doc="mean temperature")
    t_std       = Column(Float, doc="std dev of temperature")

    # Convergence
    final_l2_norm       = Column(Float)
    convergence_steps   = Column(Integer)
    converged           = Column(Boolean, default=True)
    residual_final      = Column(Float)

    # Heat flux (optional post-processing)
    max_heat_flux       = Column(Float)
    mean_heat_flux      = Column(Float)

    # Extra metrics (open-ended JSON)
    extra               = Column(JSON, default=dict)

    run = relationship("SimulationRun", back_populates="results")


class ConvergenceRecord(Base):
    """Per-timestep convergence data for plotting and analysis."""
    __tablename__ = "convergence_history"

    id       = Column(Integer, primary_key=True)
    run_id   = Column(Integer, ForeignKey("simulation_runs.id", ondelete="CASCADE"),
                      nullable=False, index=True)
    step     = Column(Integer, nullable=False)
    time     = Column(Float,   nullable=False)
    l2_norm  = Column(Float,   nullable=False)
    residual = Column(Float)
    t_max    = Column(Float)
    t_min    = Column(Float)

    run = relationship("SimulationRun", back_populates="convergence_history")

    __table_args__ = (
        UniqueConstraint("run_id", "step", name="uq_convergence_step"),
        Index("ix_convergence_run_step", "run_id", "step"),
    )


# ─── Parametric Studies ───────────────────────────────────────────────────────

class ParametricStudy(Base):
    """A group of simulation runs that form a parametric sweep."""
    __tablename__ = "parametric_studies"

    id             = Column(Integer, primary_key=True)
    study_id       = Column(String(128), nullable=False, unique=True, index=True)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    name           = Column(String(256), nullable=False)
    description    = Column(Text, default="")
    swept_parameter = Column(String(128))         # e.g. "k", "dt", "nx"
    parameter_values = Column(JSON, default=list) # list of swept values
    base_config    = Column(JSON, default=dict)
    tags           = Column(JSON, default=list)
    status         = Column(String(32), default="pending")

    study_runs = relationship("StudyRun", back_populates="study",
                               cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ParametricStudy study_id={self.study_id!r} param={self.swept_parameter}>"


class StudyRun(Base):
    """Many-to-many link between parametric studies and simulation runs."""
    __tablename__ = "study_runs"

    id         = Column(Integer, primary_key=True)
    study_id   = Column(Integer, ForeignKey("parametric_studies.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    run_db_id  = Column(Integer, ForeignKey("simulation_runs.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    param_value = Column(Float)    # the swept parameter value for this run
    order_idx   = Column(Integer)  # run order within the study

    study = relationship("ParametricStudy", back_populates="study_runs")
    run   = relationship("SimulationRun",   back_populates="study_runs")

    __table_args__ = (
        UniqueConstraint("study_id", "run_db_id", name="uq_study_run"),
    )


# ─── Agent Communication Log ──────────────────────────────────────────────────

class AgentMessage(Base):
    """Records all messages exchanged between agents (audit trail)."""
    __tablename__ = "agent_messages"

    id           = Column(Integer, primary_key=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    sender       = Column(Enum(AgentName), nullable=False, index=True)
    receiver     = Column(Enum(AgentName), nullable=False, index=True)
    msg_type     = Column(Enum(MessageType), nullable=False)
    run_id       = Column(String(128), index=True)
    study_id     = Column(String(128), index=True)
    content      = Column(JSON, default=dict)
    raw_text     = Column(Text, default="")
    processing_ms = Column(Integer)  # LLM latency in ms

    def __repr__(self):
        return (f"<AgentMessage {self.sender.value}→{self.receiver.value} "
                f"type={self.msg_type.value}>")


class AgentRunLog(Base):
    """
    Step-by-step trace of an agent's decision-making for a single task.

    One row per step (reasoning / tool_call / tool_result / final_answer).
    All rows belonging to one agent invocation share the same `task_id`.
    The `run_id` column is back-filled once the simulation run_id is known
    from a tool result, so early steps may have run_id=NULL.
    """
    __tablename__ = "agent_run_logs"

    id          = Column(Integer, primary_key=True)
    task_id     = Column(String(64),  nullable=False, index=True)
    run_id      = Column(String(128), nullable=True,  index=True)
    agent_name  = Column(String(64),  nullable=False, index=True)
    step_index  = Column(Integer,     nullable=False, default=0)
    step_type   = Column(String(32),  nullable=False)   # reasoning|tool_call|tool_result|final_answer
    content     = Column(JSON,        default=dict)
    elapsed_ms  = Column(Integer,     nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index("ix_agent_run_logs_task_step", "task_id", "step_index"),
    )

    def __repr__(self):
        return (f"<AgentRunLog task={self.task_id[:8]} step={self.step_index}"
                f" type={self.step_type}>")


class AgentSuggestion(Base):
    """Suggestions produced by the Analytics agent for new simulation runs."""
    __tablename__ = "agent_suggestions"

    id           = Column(Integer, primary_key=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    source_run_id = Column(String(128), index=True)
    rationale    = Column(Text)
    suggested_config = Column(JSON, default=dict)
    priority     = Column(Integer, default=5)     # 1=highest, 10=lowest
    accepted     = Column(Boolean)
    accepted_at  = Column(DateTime(timezone=True))
    accepted_run_id = Column(String(128))         # run_id if accepted and executed

    def __repr__(self):
        return f"<AgentSuggestion id={self.id} priority={self.priority} accepted={self.accepted}>"
