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

All metrics are computed over a single, explicitly bounded *reporting
window* (see below) so that every number in the resulting table is drawn
from the same population of runs.

Usage:
    # From the agents container (default window):
    python /app/evaluation/metrics/agent_quality.py

    # Or from host via docker compose:
    docker compose exec agents python /app/evaluation/metrics/agent_quality.py

    # Widen the window or drop it entirely:
    python agent_quality.py --until 2026-07-01
    python agent_quality.py --all
"""

from __future__ import annotations

import argparse
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

from sqlalchemy import create_engine, select, func, distinct, and_, or_, case, text
from sqlalchemy.orm import Session, sessionmaker

from database.models import (
    Base, AgentRunLog, AgentSuggestion, SimulationRun, RunStatus,
    AgentMessage, ConvergenceRecord,
)
from database.operations import get_database_url


# ─── Reporting window ────────────────────────────────────────────────────────
# The paper reports production metrics over organic FEniCSx usage. Two events
# would otherwise contaminate that population:
#
#   2026-04-20  the controlled KG ablation campaign began. Its runs (including
#               its deliberate KG On failures) are an experimental *result*
#               reported in the ablation section, not production usage.
#   2026-06-15  the SharedModules/Alsim backend was introduced, adding a second
#               solver whose runs are out of scope for a FEniCSx-only paper.
#
# Ending the window at 2026-04-17 excludes both. The date filter alone already
# rules out every Alsim run, but the backend prefix filter is applied too so
# the intent stays explicit and the script stays correct under a wider window.
WINDOW_START = os.getenv("METRICS_WINDOW_START", "")              # inclusive; "" = unbounded
WINDOW_END   = os.getenv("METRICS_WINDOW_END", "2026-04-17")      # exclusive; "" = unbounded
EXCLUDE_RUN_PREFIXES: tuple[str, ...] = ("alsim_",)               # non-FEniCSx backends


def _window_description() -> str:
    start = WINDOW_START or "(beginning)"
    end = WINDOW_END or "(now)"
    return f"{start} .. {end}"


def _run_clauses() -> list:
    """Filters scoping SimulationRun rows to the reporting window."""
    clauses = []
    if WINDOW_START:
        clauses.append(SimulationRun.created_at >= WINDOW_START)
    if WINDOW_END:
        clauses.append(SimulationRun.created_at < WINDOW_END)
    for prefix in EXCLUDE_RUN_PREFIXES:
        clauses.append(~SimulationRun.run_id.like(f"{prefix}%"))
    return clauses


def _log_clauses() -> list:
    """Filters scoping AgentRunLog rows to the reporting window.

    ``run_id`` is NULL for reasoning steps logged before the run is created,
    so those rows are kept; only steps explicitly tied to a non-FEniCSx run
    are dropped.
    """
    clauses = []
    if WINDOW_START:
        clauses.append(AgentRunLog.created_at >= WINDOW_START)
    if WINDOW_END:
        clauses.append(AgentRunLog.created_at < WINDOW_END)
    for prefix in EXCLUDE_RUN_PREFIXES:
        clauses.append(or_(AgentRunLog.run_id.is_(None),
                           ~AgentRunLog.run_id.like(f"{prefix}%")))
    return clauses


def _suggestion_clauses() -> list:
    """Filters scoping AgentSuggestion rows to the reporting window."""
    clauses = []
    if WINDOW_START:
        clauses.append(AgentSuggestion.created_at >= WINDOW_START)
    if WINDOW_END:
        clauses.append(AgentSuggestion.created_at < WINDOW_END)
    return clauses


def get_session() -> Session:
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    return sessionmaker(bind=engine)()


@dataclass
class AgentMetrics:
    timestamp: str
    window: dict
    db_stats: dict
    task_metrics: dict
    tool_metrics: dict
    suggestion_metrics: dict
    timing_metrics: dict
    orchestrator_metrics: dict


def compute_db_stats(session: Session) -> dict:
    """Basic counts from the database, scoped to the reporting window."""
    runs = _run_clauses()
    n_runs = session.scalar(select(func.count(SimulationRun.id)).where(*runs))
    n_success = session.scalar(
        select(func.count(SimulationRun.id))
        .where(*runs, SimulationRun.status == RunStatus.SUCCESS)
    )
    n_failed = session.scalar(
        select(func.count(SimulationRun.id))
        .where(*runs, SimulationRun.status == RunStatus.FAILED)
    )
    n_log_entries = session.scalar(
        select(func.count(AgentRunLog.id)).where(*_log_clauses())
    )
    n_unique_tasks = session.scalar(
        select(func.count(distinct(AgentRunLog.task_id))).where(*_log_clauses())
    )
    n_suggestions = session.scalar(
        select(func.count(AgentSuggestion.id)).where(*_suggestion_clauses())
    )

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
        .where(*_log_clauses())
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


def _tool_name(content) -> str:
    """Extract the tool name from an AgentRunLog content payload.

    Tool-call rows are logged under the key ``tool``; some result payloads
    and older rows use ``tool_name`` or ``name``.
    """
    if not isinstance(content, dict):
        return "unknown"
    return (content.get("tool")
            or content.get("tool_name")
            or content.get("name")
            or "unknown")


def _result_failed(payload) -> bool:
    """Decide whether a logged tool result represents a failure."""
    if isinstance(payload, dict):
        if "error" in payload:
            return True
        if str(payload.get("status", "")).lower() in ("failed", "error"):
            return True
        return False
    text = str(payload).lower()
    return '"error"' in text or "'error'" in text


def compute_tool_metrics(session: Session) -> dict:
    """Tool invocation success/failure rates.

    ``tool_result`` rows carry only a ``tool_call_id`` and the payload, not the
    tool name, so each result is matched back to the ``tool_call`` that
    immediately precedes it in the same task (``step_index - 1``).
    """
    rows = session.execute(
        select(AgentRunLog.task_id, AgentRunLog.step_index,
               AgentRunLog.step_type, AgentRunLog.content)
        .where(*_log_clauses(),
               AgentRunLog.step_type.in_(("tool_call", "tool_result")))
    ).all()

    call_at: dict[tuple[str, int], str] = {}
    for task_id, step_index, step_type, content in rows:
        if step_type == "tool_call":
            call_at[(task_id, step_index)] = _tool_name(content)

    tool_stats = defaultdict(lambda: {"calls": 0, "successes": 0, "failures": 0,
                                      "ungraded": 0})

    for name in call_at.values():
        tool_stats[name]["calls"] += 1

    for task_id, step_index, step_type, content in rows:
        if step_type != "tool_result":
            continue
        name = call_at.get((task_id, step_index - 1))
        if name is None:
            tool_stats["unmatched"]["ungraded"] += 1
            continue
        payload = content.get("result") if isinstance(content, dict) else content
        if _result_failed(payload):
            tool_stats[name]["failures"] += 1
        else:
            tool_stats[name]["successes"] += 1

    for stats in tool_stats.values():
        graded = stats["successes"] + stats["failures"]
        stats["success_rate"] = (stats["successes"] / graded) if graded else None

    return dict(tool_stats)


def compute_suggestion_metrics(session: Session) -> dict:
    """Analytics agent suggestion acceptance and impact."""
    sugg = _suggestion_clauses()
    total = session.scalar(select(func.count(AgentSuggestion.id)).where(*sugg))
    accepted = session.scalar(
        select(func.count(AgentSuggestion.id))
        .where(*sugg, AgentSuggestion.accepted == True)
    )
    rejected = session.scalar(
        select(func.count(AgentSuggestion.id))
        .where(*sugg, AgentSuggestion.accepted == False)
    )
    pending = total - (accepted or 0) - (rejected or 0)

    # Priority distribution
    priority_rows = session.execute(
        select(
            AgentSuggestion.priority,
            func.count(AgentSuggestion.id),
        )
        .where(*sugg)
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
        .where(*_log_clauses(), AgentRunLog.elapsed_ms.isnot(None))
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
        .where(*_run_clauses(), SimulationRun.wall_time.isnot(None))
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
        .where(*_log_clauses(), AgentRunLog.agent_name == "orchestrator")
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
    """Fraction of simulation tasks that succeeded without a retry.

    A task counts as first-try successful when it invoked ``run_simulation``
    exactly once, never invoked ``debug_simulation``, and the resulting run
    finished with SUCCESS.

    Retries are detected from tool-call logs rather than from distinct
    ``run_id`` values.  ``backfill_task_run_id`` rewrites *every* log row of a
    task with the most recently known run_id, so a task that retried collapses
    to a single run_id and is indistinguishable from a clean run by that route.
    Counting ``run_simulation`` invocations recovers the true attempt count.
    """
    tool_rows = session.execute(
        select(AgentRunLog.task_id, AgentRunLog.content)
        .where(*_log_clauses(),
               AgentRunLog.agent_name == "simulation",
               AgentRunLog.step_type == "tool_call")
    ).all()

    launches: dict[str, int] = defaultdict(int)
    debugged: set[str] = set()
    for task_id, content in tool_rows:
        name = _tool_name(content)
        if name == "run_simulation":
            launches[task_id] += 1
        elif name == "debug_simulation":
            debugged.add(task_id)

    # run_id is reliable for single-attempt tasks: the backfill overwrote it
    # with the only run the task ever produced.
    task_run = {
        row.task_id: row.run_id
        for row in session.execute(
            select(AgentRunLog.task_id, AgentRunLog.run_id)
            .where(*_log_clauses(),
                   AgentRunLog.agent_name == "simulation",
                   AgentRunLog.run_id.isnot(None))
            .distinct()
        ).all()
    }

    succeeded_runs = {
        rid for (rid,) in session.execute(
            select(SimulationRun.run_id)
            .where(*_run_clauses(), SimulationRun.status == RunStatus.SUCCESS)
        ).all()
    }

    n_tasks = len(launches)
    single_attempt = {t for t, n in launches.items()
                      if n == 1 and t not in debugged}
    retried = n_tasks - len(single_attempt)
    n_first_try_success = sum(
        1 for t in single_attempt if task_run.get(t) in succeeded_runs
    )

    return {
        "total_sim_tasks": n_tasks,
        "single_attempt_tasks": len(single_attempt),
        "retried_tasks": retried,
        "first_try_successes": n_first_try_success,
        "first_try_success_rate": n_first_try_success / max(n_tasks, 1),
        "retry_rate": retried / max(n_tasks, 1),
        "attempt_detection": "run_simulation tool-call count (run_id is overwritten by backfill)",
    }


def run_all_metrics(output_path: str | None = None) -> AgentMetrics:
    """Compute all metrics and return a structured result."""
    session = get_session()

    print(f"\n{'='*60}")
    print(f"  AGENT DECISION QUALITY METRICS")
    print(f"{'='*60}")
    print(f"  Reporting window: {_window_description()}")
    print(f"  Excluded backends: {', '.join(EXCLUDE_RUN_PREFIXES) or 'none'}\n")

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
        window={
            "start": WINDOW_START or None,
            "end": WINDOW_END or None,
            "excluded_run_prefixes": list(EXCLUDE_RUN_PREFIXES),
            "rationale": (
                "Organic FEniCSx usage only: ends before the controlled KG "
                "ablation campaign (2026-04-20) and the SharedModules backend "
                "(2026-06-15), so production metrics exclude experimental runs "
                "and non-FEniCSx solvers."
            ),
        },
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
    for tool_name, stats in sorted(tool_metrics.items(),
                                   key=lambda kv: -kv[1]["calls"]):
        rate = stats["success_rate"]
        rate_s = f"{rate:.2f}" if rate is not None else "n/a"
        print(f"    {tool_name}: {stats['calls']} calls, success_rate={rate_s}")

    print(f"\n  ─── Suggestions ───")
    print(f"    Total: {suggestion_metrics['total']}, "
          f"Accepted: {suggestion_metrics['accepted']}, "
          f"Rate: {suggestion_metrics['acceptance_rate']:.2f}")

    print(f"\n  ─── First-Try Success ───")
    print(f"    {first_try['first_try_successes']}/{first_try['total_sim_tasks']} = "
          f"{first_try['first_try_success_rate']:.3f}")
    print(f"    retried: {first_try['retried_tasks']} "
          f"({first_try['retry_rate']:.1%})")

    print(f"\n  ─── Simulation Timing ───")
    st = timing_metrics["simulation_wall_time"]
    print(f"    Avg: {st['avg_s']:.2f}s  Min: {st['min_s']:.2f}s  "
          f"Max: {st['max_s']:.2f}s  (n={st['n_runs']})")

    # Save results
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path) if output_path else output_dir / "agent_metrics.json"
    with open(output_file, "w") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    return metrics


def main() -> None:
    global WINDOW_START, WINDOW_END, EXCLUDE_RUN_PREFIXES

    parser = argparse.ArgumentParser(
        description="Compute agent decision quality metrics over a bounded window.",
    )
    parser.add_argument("--since", default=WINDOW_START, metavar="YYYY-MM-DD",
                        help="Window start, inclusive (default: unbounded)")
    parser.add_argument("--until", default=WINDOW_END, metavar="YYYY-MM-DD",
                        help=f"Window end, exclusive (default: {WINDOW_END})")
    parser.add_argument("--all", action="store_true",
                        help="Ignore the window and include every run and backend")
    parser.add_argument("-o", "--output", default=None, metavar="PATH",
                        help="Write JSON here instead of results/agent_metrics.json")
    args = parser.parse_args()

    if args.all:
        WINDOW_START, WINDOW_END, EXCLUDE_RUN_PREFIXES = "", "", ()
    else:
        WINDOW_START, WINDOW_END = args.since, args.until

    run_all_metrics(output_path=args.output)


if __name__ == "__main__":
    main()
