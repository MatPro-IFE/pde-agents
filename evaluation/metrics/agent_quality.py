#!/usr/bin/env python3
"""
Agent decision quality metrics — mines the PostgreSQL database for
quantitative evidence of agent performance.

Metrics computed:
  1. Task completion rate (per agent, per difficulty)
  2. Average reasoning steps per task (lower = more efficient)
  3. Tool call accuracy (successful vs failed tool invocations)
  4. First-try success rate (simulation succeeded without debug/retry)
  5. Config warning adoption rate (agent heeded KG warnings)
  6. Suggestion acceptance rate (analytics suggestions that led to runs)
  7. Wall-time breakdown (LLM inference vs tool execution vs total)
  8. Orchestrator routing efficiency (iterations per task)

Usage:
    # From the agents container:
    python /app/evaluation/metrics/agent_quality.py

    # Or from host via docker compose:
    docker compose exec agents python /app/evaluation/metrics/agent_quality.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

# Allow running from both host (evaluation/) and container (/app)
sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import create_engine, select, func, distinct, and_, case, text
from sqlalchemy.orm import Session, sessionmaker

from database.models import (
    Base, AgentRunLog, AgentSuggestion, SimulationRun, RunStatus,
    AgentMessage, ConvergenceRecord,
)
from database.operations import get_database_url


def get_session() -> Session:
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    return sessionmaker(bind=engine)()


@dataclass
class AgentMetrics:
    timestamp: str
    db_stats: dict
    task_metrics: dict
    tool_metrics: dict
    suggestion_metrics: dict
    timing_metrics: dict
    orchestrator_metrics: dict


def compute_db_stats(session: Session) -> dict:
    """Basic counts from the database."""
    n_runs = session.scalar(select(func.count(SimulationRun.id)))
    n_success = session.scalar(
        select(func.count(SimulationRun.id))
        .where(SimulationRun.status == RunStatus.SUCCESS)
    )
    n_failed = session.scalar(
        select(func.count(SimulationRun.id))
        .where(SimulationRun.status == RunStatus.FAILED)
    )
    n_log_entries = session.scalar(select(func.count(AgentRunLog.id)))
    n_unique_tasks = session.scalar(
        select(func.count(distinct(AgentRunLog.task_id)))
    )
    n_suggestions = session.scalar(select(func.count(AgentSuggestion.id)))

    return {
        "total_runs": n_runs,
        "successful_runs": n_success,
        "failed_runs": n_failed,
        "overall_success_rate": n_success / max(n_runs, 1),
        "total_log_entries": n_log_entries,
        "unique_tasks": n_unique_tasks,
        "total_suggestions": n_suggestions,
    }


def compute_task_metrics(session: Session) -> dict:
    """Per-agent task performance metrics."""
    results = {}

    # Steps per task grouped by agent
    rows = session.execute(
        select(
            AgentRunLog.agent_name,
            AgentRunLog.task_id,
            func.count(AgentRunLog.id).label("n_steps"),
            func.max(AgentRunLog.step_index).label("max_step"),
        )
        .group_by(AgentRunLog.agent_name, AgentRunLog.task_id)
    ).all()

    agent_tasks = defaultdict(list)
    for row in rows:
        agent_tasks[row.agent_name].append({
            "task_id": row.task_id,
            "n_steps": row.n_steps,
            "max_step": row.max_step,
        })

    for agent_name, tasks in agent_tasks.items():
        steps = [t["n_steps"] for t in tasks]
        results[agent_name] = {
            "n_tasks": len(tasks),
            "avg_steps_per_task": sum(steps) / len(steps) if steps else 0,
            "min_steps": min(steps) if steps else 0,
            "max_steps": max(steps) if steps else 0,
            "median_steps": sorted(steps)[len(steps) // 2] if steps else 0,
        }

    return results


def compute_tool_metrics(session: Session) -> dict:
    """Tool invocation success/failure rates."""
    tool_calls = session.execute(
        select(AgentRunLog.content, AgentRunLog.agent_name)
        .where(AgentRunLog.step_type == "tool_call")
    ).all()

    tool_results = session.execute(
        select(AgentRunLog.content, AgentRunLog.agent_name)
        .where(AgentRunLog.step_type == "tool_result")
    ).all()

    tool_stats = defaultdict(lambda: {"calls": 0, "successes": 0, "failures": 0})

    for row in tool_calls:
        content = row.content if isinstance(row.content, dict) else {}
        tool_name = content.get("tool_name", content.get("name", "unknown"))
        tool_stats[tool_name]["calls"] += 1

    for row in tool_results:
        content = row.content if isinstance(row.content, dict) else {}
        tool_name = content.get("tool_name", content.get("name", "unknown"))
        result_text = str(content.get("result", content.get("output", "")))
        if "error" in result_text.lower() or "failed" in result_text.lower():
            tool_stats[tool_name]["failures"] += 1
        else:
            tool_stats[tool_name]["successes"] += 1

    for tool_name, stats in tool_stats.items():
        total = stats["calls"]
        stats["success_rate"] = stats["successes"] / max(total, 1)

    return dict(tool_stats)


def compute_suggestion_metrics(session: Session) -> dict:
    """Analytics agent suggestion acceptance and impact."""
    total = session.scalar(select(func.count(AgentSuggestion.id)))
    accepted = session.scalar(
        select(func.count(AgentSuggestion.id))
        .where(AgentSuggestion.accepted == True)
    )
    rejected = session.scalar(
        select(func.count(AgentSuggestion.id))
        .where(AgentSuggestion.accepted == False)
    )
    pending = total - (accepted or 0) - (rejected or 0)

    # Priority distribution
    priority_rows = session.execute(
        select(
            AgentSuggestion.priority,
            func.count(AgentSuggestion.id),
        )
        .group_by(AgentSuggestion.priority)
        .order_by(AgentSuggestion.priority)
    ).all()

    return {
        "total": total,
        "accepted": accepted or 0,
        "rejected": rejected or 0,
        "pending": pending,
        "acceptance_rate": (accepted or 0) / max(total, 1),
        "priority_distribution": {row[0]: row[1] for row in priority_rows},
    }


def compute_timing_metrics(session: Session) -> dict:
    """Wall-time statistics from agent logs and simulation runs."""
    # Agent step timings
    timing_rows = session.execute(
        select(
            AgentRunLog.agent_name,
            AgentRunLog.step_type,
            func.avg(AgentRunLog.elapsed_ms).label("avg_ms"),
            func.min(AgentRunLog.elapsed_ms).label("min_ms"),
            func.max(AgentRunLog.elapsed_ms).label("max_ms"),
            func.count(AgentRunLog.id).label("n"),
        )
        .where(AgentRunLog.elapsed_ms.isnot(None))
        .group_by(AgentRunLog.agent_name, AgentRunLog.step_type)
    ).all()

    agent_timing = defaultdict(dict)
    for row in timing_rows:
        agent_timing[row.agent_name][row.step_type] = {
            "avg_ms": float(row.avg_ms) if row.avg_ms else 0,
            "min_ms": int(row.min_ms) if row.min_ms else 0,
            "max_ms": int(row.max_ms) if row.max_ms else 0,
            "n_samples": row.n,
        }

    # Simulation wall times
    sim_timing = session.execute(
        select(
            func.avg(SimulationRun.wall_time).label("avg"),
            func.min(SimulationRun.wall_time).label("min"),
            func.max(SimulationRun.wall_time).label("max"),
            func.count(SimulationRun.id).label("n"),
        )
        .where(SimulationRun.wall_time.isnot(None))
    ).one()

    return {
        "agent_step_timing": dict(agent_timing),
        "simulation_wall_time": {
            "avg_s": float(sim_timing.avg) if sim_timing.avg else 0,
            "min_s": float(sim_timing.min) if sim_timing.min else 0,
            "max_s": float(sim_timing.max) if sim_timing.max else 0,
            "n_runs": sim_timing.n,
        },
    }


def compute_orchestrator_metrics(session: Session) -> dict:
    """Orchestrator iteration counts and routing patterns."""
    # Tasks that went through the orchestrator
    orch_tasks = session.execute(
        select(
            AgentRunLog.task_id,
            func.count(AgentRunLog.id).label("n_steps"),
        )
        .where(AgentRunLog.agent_name == "orchestrator")
        .group_by(AgentRunLog.task_id)
    ).all()

    if not orch_tasks:
        # No orchestrator logs; compute from inter-agent messages
        msg_count = session.scalar(select(func.count(AgentMessage.id)))
        return {
            "n_orchestrated_tasks": 0,
            "total_agent_messages": msg_count or 0,
            "note": "No orchestrator task logs found; agent messages available",
        }

    iterations = [row.n_steps for row in orch_tasks]
    return {
        "n_orchestrated_tasks": len(orch_tasks),
        "avg_iterations": sum(iterations) / len(iterations),
        "min_iterations": min(iterations),
        "max_iterations": max(iterations),
    }


def compute_first_try_success(session: Session) -> dict:
    """Fraction of simulation tasks that succeeded on first attempt (no retry)."""
    # Find tasks that produced a run_id
    task_run_pairs = session.execute(
        select(
            AgentRunLog.task_id,
            AgentRunLog.run_id,
        )
        .where(
            and_(
                AgentRunLog.agent_name == "simulation",
                AgentRunLog.run_id.isnot(None),
            )
        )
        .distinct()
    ).all()

    task_runs = defaultdict(set)
    for row in task_run_pairs:
        task_runs[row.task_id].add(row.run_id)

    n_tasks = len(task_runs)
    n_single_run = sum(1 for runs in task_runs.values() if len(runs) == 1)

    # Check if those single-run tasks succeeded
    n_first_try_success = 0
    for task_id, run_ids in task_runs.items():
        if len(run_ids) == 1:
            run_id = list(run_ids)[0]
            run = session.scalar(
                select(SimulationRun)
                .where(SimulationRun.run_id == run_id)
            )
            if run and run.status == RunStatus.SUCCESS:
                n_first_try_success += 1

    return {
        "total_sim_tasks": n_tasks,
        "single_attempt_tasks": n_single_run,
        "first_try_successes": n_first_try_success,
        "first_try_success_rate": n_first_try_success / max(n_tasks, 1),
    }


def run_all_metrics() -> AgentMetrics:
    """Compute all metrics and return a structured result."""
    session = get_session()

    print(f"\n{'='*60}")
    print(f"  AGENT DECISION QUALITY METRICS")
    print(f"{'='*60}\n")

    print("  Computing database stats...")
    db_stats = compute_db_stats(session)

    print("  Computing task metrics...")
    task_metrics = compute_task_metrics(session)

    print("  Computing tool metrics...")
    tool_metrics = compute_tool_metrics(session)

    print("  Computing suggestion metrics...")
    suggestion_metrics = compute_suggestion_metrics(session)

    print("  Computing timing metrics...")
    timing_metrics = compute_timing_metrics(session)

    print("  Computing orchestrator metrics...")
    orchestrator_metrics = compute_orchestrator_metrics(session)

    print("  Computing first-try success rate...")
    first_try = compute_first_try_success(session)

    session.close()

    metrics = AgentMetrics(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        db_stats=db_stats,
        task_metrics=task_metrics,
        tool_metrics=tool_metrics,
        suggestion_metrics=suggestion_metrics,
        timing_metrics=timing_metrics,
        orchestrator_metrics={**orchestrator_metrics, **first_try},
    )

    # Print summary
    print(f"\n  ─── Database ───")
    for k, v in db_stats.items():
        print(f"    {k}: {v}")

    print(f"\n  ─── Per-Agent Task Stats ───")
    for agent, stats in task_metrics.items():
        print(f"    {agent}: {stats['n_tasks']} tasks, "
              f"avg {stats['avg_steps_per_task']:.1f} steps/task")

    print(f"\n  ─── Tool Usage ───")
    for tool_name, stats in sorted(tool_metrics.items()):
        print(f"    {tool_name}: {stats['calls']} calls, "
              f"success_rate={stats['success_rate']:.2f}")

    print(f"\n  ─── Suggestions ───")
    print(f"    Total: {suggestion_metrics['total']}, "
          f"Accepted: {suggestion_metrics['accepted']}, "
          f"Rate: {suggestion_metrics['acceptance_rate']:.2f}")

    print(f"\n  ─── First-Try Success ───")
    print(f"    {first_try['first_try_successes']}/{first_try['total_sim_tasks']} = "
          f"{first_try['first_try_success_rate']:.2f}")

    print(f"\n  ─── Simulation Timing ───")
    st = timing_metrics["simulation_wall_time"]
    print(f"    Avg: {st['avg_s']:.2f}s  Min: {st['min_s']:.2f}s  "
          f"Max: {st['max_s']:.2f}s  (n={st['n_runs']})")

    # Save results
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "agent_metrics.json"
    with open(output_file, "w") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    return metrics


if __name__ == "__main__":
    run_all_metrics()
