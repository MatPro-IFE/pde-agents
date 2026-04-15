#!/usr/bin/env python3
"""
Ablation study runner: KG-augmented vs baseline agents.

Two execution modes:

  --direct (recommended, default inside the container)
    Instantiates SimulationAgent directly and calls agent.run().
    Avoids uvicorn/asyncio thread-pool issues with LangGraph sync invoke.

  --api-mode
    Sends each benchmark task to the agents HTTP API:
      POST /simulate              (KG on)
      POST /simulate?disable_kg   (KG off)

Measures:
  - Success rate (simulation completed without error)
  - Config quality (material properties within expected ranges)
  - Wall time per task
  - Number of agent iterations (reasoning steps)
  - First-try success rate (no debug/retry needed)

Usage inside the container:
    docker compose exec agents python /app/evaluation/ablation/run_ablation.py

Usage from host (API mode):
    python evaluation/ablation/run_ablation.py --api-mode --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation.benchmark_tasks import ABLATION_TASKS, NOVIDIUM_TASKS

DEFAULT_API_URL = "http://localhost:8000"
TIMEOUT = 600  # 10 minutes per task


@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    mode: str       # "kg_on" or "kg_off"
    success: bool
    wall_time_s: float
    agent_iterations: int
    run_id: str | None
    config_produced: dict
    config_quality_score: float  # 0.0–1.0
    first_try_success: bool
    error_message: str = ""
    raw_response: dict = field(default_factory=dict)


def score_config(produced_config: dict, ground_truth: dict) -> float:
    """Score how well the agent's config matches ground truth expectations.

    Returns 0.0–1.0 where 1.0 = perfect match of all checkable fields.
    """
    checks = 0
    passes = 0

    for field_name in ("dim", "nx", "ny", "nz", "theta", "source"):
        if field_name in ground_truth:
            checks += 1
            if produced_config.get(field_name) == ground_truth[field_name]:
                passes += 1

    for range_field in ("k_range", "rho_range", "cp_range"):
        if range_field in ground_truth:
            base = range_field.replace("_range", "")
            val = produced_config.get(base)
            checks += 1
            if val is not None:
                lo, hi = ground_truth[range_field]
                if lo <= val <= hi:
                    passes += 1

    for range_field in ("T_max_range", "T_min_range"):
        if range_field in ground_truth:
            checks += 1
            # Scored from the simulation result, not just config

    if ground_truth.get("has_robin_bc"):
        checks += 1
        bcs = produced_config.get("bcs", [])
        if any(bc.get("type") == "robin" for bc in bcs):
            passes += 1

    return passes / max(checks, 1)


def extract_config_from_response(response: dict) -> dict:
    """Extract the simulation config the agent produced from the response.

    The real API wraps agent output as: {request_id, status, result, error}
    where result = {answer, iterations, tool_calls_log, task_id}.
    We mine tool_calls_log for any run_simulation call to get the config.
    """
    result = response.get("result") or response
    if not isinstance(result, dict):
        return {}

    # Mine tool call log for run_simulation arguments
    tool_calls_log = result.get("tool_calls_log", [])
    for entry in tool_calls_log:
        if isinstance(entry, dict):
            tool_name = entry.get("tool_name", entry.get("name", ""))
            if tool_name in ("run_simulation", "validate_config"):
                args = entry.get("args", entry.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        continue
                if isinstance(args, dict):
                    cfg = args.get("config", args)
                    if isinstance(cfg, dict) and any(k in cfg for k in
                                                     ("nx", "k", "dim", "bcs")):
                        return cfg

    # Fallback: try answer text
    answer = result.get("answer", "")
    if isinstance(answer, dict):
        return answer.get("config", answer)
    try:
        parsed = json.loads(str(answer))
        if isinstance(parsed, dict):
            return parsed.get("config", parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def count_iterations(response: dict) -> int:
    """Extract the number of agent reasoning iterations."""
    result = response.get("result") or response
    if isinstance(result, dict):
        return result.get("iterations", result.get("total_iterations", 1))
    return response.get("total_iterations", 1)


def run_task_direct(task: dict, disable_kg: bool,
                    agent=None) -> TaskResult:
    """Run a benchmark task by calling a reusable pre-warmed agent directly."""
    mode = "kg_off" if disable_kg else "kg_on"
    print(f"  [{task['id']}] ({task['difficulty']}) mode={mode} ...", end=" ", flush=True)

    t0 = time.perf_counter()
    try:
        # Retry up to 3 times; qwen2.5-coder occasionally outputs a non-tool
        # first response when context is cold (LangGraph / Ollama warm-up issue).
        MAX_TRIES = 3
        result = None
        for attempt in range(MAX_TRIES):
            result = agent.run(task["description"])
            iters = result.get("iterations", 1)
            if iters > 1 or result.get("run_id"):
                break  # real execution happened
            if attempt < MAX_TRIES - 1:
                print(f"(retry {attempt+1}) ", end="", flush=True)

        wall = time.perf_counter() - t0

        answer = result.get("answer", "")
        success = (result.get("status") == "done" and
                   "success" in str(answer).lower() and
                   "fail" not in str(answer).lower())
        iterations = result.get("iterations", 1)
        run_id = result.get("run_id")

        # Mine tool_calls_log for the config used
        config_produced = {}
        for entry in result.get("tool_calls_log", []):
            if isinstance(entry, dict) and entry.get("has_tool_calls"):
                pass  # config extraction done below
        config_produced = extract_config_from_direct_result(result)

        quality = score_config(config_produced, task["ground_truth"])
        first_try = iterations <= 12 and success

        status = "OK" if success else "FAIL"
        print(f"{status}  quality={quality:.2f}  iter={iterations}  run_id={run_id}  ({wall:.1f}s)")

        return TaskResult(
            task_id=task["id"], difficulty=task["difficulty"],
            mode=mode, success=success, wall_time_s=wall,
            agent_iterations=iterations, run_id=run_id,
            config_produced=config_produced,
            config_quality_score=quality,
            first_try_success=first_try,
            raw_response=result,
        )
    except Exception as e:
        wall = time.perf_counter() - t0
        print(f"ERROR ({wall:.1f}s): {str(e)[:80]}")
        return TaskResult(
            task_id=task["id"], difficulty=task["difficulty"],
            mode=mode, success=False, wall_time_s=wall,
            agent_iterations=0, run_id=None,
            config_produced={}, config_quality_score=0.0,
            first_try_success=False, error_message=str(e)[:200],
        )


def extract_config_from_direct_result(result: dict) -> dict:
    """Extract simulation config from a direct agent.run() result dict."""
    # Try mining agent's DB log for the run's config
    run_id = result.get("run_id")
    if run_id:
        try:
            sys.path.insert(0, "/app")
            from database.operations import get_run, get_session_factory
            from sqlalchemy.orm import Session
            sf = get_session_factory()
            with sf() as session:
                run = session.execute(
                    __import__("sqlalchemy").select(
                        __import__("database.models", fromlist=["SimulationRun"]).SimulationRun
                    ).where(
                        __import__("database.models", fromlist=["SimulationRun"])
                        .SimulationRun.run_id == run_id
                    )
                ).scalar_one_or_none()
                if run and run.config_json:
                    cfg = run.config_json if isinstance(run.config_json, dict) else {}
                    return cfg
        except Exception:
            pass

    # Fallback: parse answer text
    answer = result.get("answer", "")
    try:
        parsed = json.loads(str(answer))
        if isinstance(parsed, dict):
            return parsed.get("config", parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def run_task(task: dict, api_url: str, disable_kg: bool) -> TaskResult:
    """Run a single benchmark task against the API."""
    mode = "kg_off" if disable_kg else "kg_on"
    print(f"  [{task['id']}] ({task['difficulty']}) mode={mode} ...", end=" ", flush=True)

    payload = {
        "description": task["description"],
        "config": {},
    }

    params = {}
    if disable_kg:
        params["disable_kg"] = "true"

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{api_url}/simulate",
            json=payload,
            params=params,
            timeout=TIMEOUT,
        )
        wall = time.perf_counter() - t0

        if resp.status_code != 200:
            print(f"HTTP {resp.status_code} ({wall:.1f}s)")
            return TaskResult(
                task_id=task["id"], difficulty=task["difficulty"],
                mode=mode, success=False, wall_time_s=wall,
                agent_iterations=0, run_id=None,
                config_produced={}, config_quality_score=0.0,
                first_try_success=False,
                error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
    except requests.Timeout:
        wall = time.perf_counter() - t0
        print(f"TIMEOUT ({wall:.1f}s)")
        return TaskResult(
            task_id=task["id"], difficulty=task["difficulty"],
            mode=mode, success=False, wall_time_s=wall,
            agent_iterations=0, run_id=None,
            config_produced={}, config_quality_score=0.0,
            first_try_success=False, error_message="Timeout",
        )
    except requests.ConnectionError as e:
        print(f"CONNECTION ERROR")
        return TaskResult(
            task_id=task["id"], difficulty=task["difficulty"],
            mode=mode, success=False, wall_time_s=0,
            agent_iterations=0, run_id=None,
            config_produced={}, config_quality_score=0.0,
            first_try_success=False, error_message=str(e)[:200],
        )

    # Parse results - API returns {request_id, status, result, error}
    api_status = data.get("status", "")
    result_payload = data.get("result") or {}
    answer = result_payload.get("answer", "") if isinstance(result_payload, dict) else str(result_payload)
    error_msg = data.get("error", "")
    success = (api_status == "success" and
               "success" in str(answer).lower() and
               not error_msg)
    config_produced = extract_config_from_response(data)
    iterations = count_iterations(data)
    run_id = (result_payload.get("run_id") if isinstance(result_payload, dict) else None)

    quality = score_config(config_produced, task["ground_truth"])
    # Agent typically uses ~7-10 steps for a successful first-try run;
    # "first try" means no debug/retry cycle (under ~12 steps)
    first_try = iterations <= 12 and success

    status = "OK" if success else "FAIL"
    print(f"{status}  quality={quality:.2f}  iter={iterations}  ({wall:.1f}s)")

    return TaskResult(
        task_id=task["id"], difficulty=task["difficulty"],
        mode=mode, success=success, wall_time_s=wall,
        agent_iterations=iterations, run_id=run_id,
        config_produced=config_produced,
        config_quality_score=quality,
        first_try_success=first_try,
        raw_response=data,
    )


def _make_agent(disable_kg: bool = False, smart_kg: bool = False):
    """Create and warm-up a SimulationAgent by making one lightweight call."""
    try:
        from agents.simulation_agent import SimulationAgent
    except ImportError:
        sys.path.insert(0, "/app")
        from agents.simulation_agent import SimulationAgent

    agent = SimulationAgent(disable_kg=disable_kg, smart_kg=smart_kg)
    print("  Warming up agent (running a quick sim)...", end=" ", flush=True)
    warmup = agent.run(
        "Run a minimal 2D heat equation: left T=0, right T=1, "
        "4x4 mesh, k=1, rho=1, cp=1, t_end=0.01, dt=0.005. "
        "Just run it without extensive checks."
    )
    print(f"done (iter={warmup.get('iterations',1)}, run_id={warmup.get('run_id')})")
    return agent


def aggregate(results: list[TaskResult]) -> dict:
    """Compute aggregate metrics for a set of task results."""
    n = len(results)
    successes = [r for r in results if r.success]
    return {
        "n_tasks": n,
        "success_rate": len(successes) / n if n else 0,
        "first_try_rate": sum(1 for r in results if r.first_try_success) / n if n else 0,
        "avg_quality": sum(r.config_quality_score for r in results) / n if n else 0,
        "avg_iterations": sum(r.agent_iterations for r in results) / n if n else 0,
        "avg_wall_time": sum(r.wall_time_s for r in results) / n if n else 0,
        "by_difficulty": {
            diff: {
                "n": len(subset := [r for r in results if r.difficulty == diff]),
                "success_rate": sum(1 for r in subset if r.success) / max(len(subset), 1),
                "avg_quality": sum(r.config_quality_score for r in subset) / max(len(subset), 1),
            }
            for diff in ("easy", "medium", "hard", "novel")
            if any(r.difficulty == diff for r in results)
        },
    }


def run_ablation(api_url: str, tasks: list[dict] | None = None,
                 direct: bool = True, include_smart: bool = False) -> dict:
    """Run the ablation study and return structured results.

    When include_smart=True, runs three conditions:
      1. KG On  (mandatory KG-first)
      2. KG Off (no KG)
      3. KG Smart (warm-start + lazy conditional KG)
    """
    tasks = tasks or ABLATION_TASKS
    results_kg_on = []
    results_kg_off = []
    results_kg_smart = []
    mode_label = "direct (in-process)" if direct else f"API ({api_url})"
    n_conditions = 3 if include_smart else 2

    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY: {n_conditions}-way KG comparison")
    print(f"  Tasks: {len(tasks)}  |  Mode: {mode_label}")
    print(f"{'='*70}")

    if direct:
        print("\n─── Initialising agents ───")
        print("  KG ON  →", end=" ")
        agent_kg_on = _make_agent(disable_kg=False)
        print("  KG OFF →", end=" ")
        agent_kg_off = _make_agent(disable_kg=True)
        if include_smart:
            print("  KG SMART →", end=" ")
            agent_kg_smart = _make_agent(smart_kg=True)

    # Phase 1: KG enabled (mandatory)
    print(f"\n─── Phase 1: Knowledge Graph ENABLED (mandatory) ───")
    for task in tasks:
        if direct:
            result = run_task_direct(task, disable_kg=False, agent=agent_kg_on)
        else:
            result = run_task(task, api_url, disable_kg=False)
        results_kg_on.append(result)

    # Phase 2: KG disabled
    print(f"\n─── Phase 2: Knowledge Graph DISABLED ───")
    for task in tasks:
        if direct:
            result = run_task_direct(task, disable_kg=True, agent=agent_kg_off)
        else:
            result = run_task(task, api_url, disable_kg=True)
        results_kg_off.append(result)

    # Phase 3: KG Smart (warm-start + lazy)
    if include_smart:
        print(f"\n─── Phase 3: Knowledge Graph SMART (warm-start + lazy) ───")
        for task in tasks:
            if direct:
                r = run_task_direct(task, disable_kg=False, agent=agent_kg_smart)
                r.mode = "kg_smart"
                results_kg_smart.append(r)
            else:
                r = run_task(task, api_url, disable_kg=False)
                r.mode = "kg_smart"
                results_kg_smart.append(r)

    agg_on = aggregate(results_kg_on)
    agg_off = aggregate(results_kg_off)
    agg_smart = aggregate(results_kg_smart) if include_smart else None

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Metric':<28s}  {'KG On':>8s}  {'KG Off':>8s}"
    if include_smart:
        header += f"  {'KG Smart':>9s}"
    print(header)
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}" + (f"  {'-'*9}" if include_smart else ""))

    for metric in ("success_rate", "first_try_rate", "avg_quality",
                    "avg_iterations", "avg_wall_time"):
        v_on = agg_on[metric]
        v_off = agg_off[metric]
        fmt = ".2f" if "rate" in metric or "quality" in metric else ".1f"
        line = f"  {metric:<28s}  {v_on:>8{fmt}}  {v_off:>8{fmt}}"
        if include_smart and agg_smart:
            v_smart = agg_smart[metric]
            line += f"  {v_smart:>9{fmt}}"
        print(line)

    print(f"\n  By difficulty (success rate):")
    for diff in ("easy", "medium", "hard", "novel"):
        if diff not in agg_on["by_difficulty"]:
            continue
        on_sr = agg_on["by_difficulty"][diff]["success_rate"]
        off_sr = agg_off["by_difficulty"][diff]["success_rate"]
        line = f"    {diff:<10s}  KG On: {on_sr:.2f}  KG Off: {off_sr:.2f}"
        if include_smart and agg_smart:
            smart_sr = agg_smart["by_difficulty"].get(diff, {}).get("success_rate", 0)
            line += f"  KG Smart: {smart_sr:.2f}"
        print(line)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_tasks": len(tasks),
        "kg_on": {
            "aggregate": agg_on,
            "tasks": [asdict(r) for r in results_kg_on],
        },
        "kg_off": {
            "aggregate": agg_off,
            "tasks": [asdict(r) for r in results_kg_off],
        },
    }
    if include_smart:
        summary["kg_smart"] = {
            "aggregate": agg_smart,
            "tasks": [asdict(r) for r in results_kg_smart],
        }

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run KG ablation study")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--tasks", nargs="*", help="Task IDs to run (default: all)")
    parser.add_argument("--api-mode", action="store_true",
                        help="Use HTTP API instead of direct in-process agent calls")
    parser.add_argument("--include-smart", action="store_true",
                        help="Include KG Smart (warm-start + lazy) as a third condition")
    parser.add_argument("--smart-only", action="store_true",
                        help="Run only the KG Smart condition (skip KG On and KG Off)")
    parser.add_argument("--novidium", action="store_true",
                        help="Run only the novel-material (Novidium) tasks (G1-G3)")
    args = parser.parse_args()

    if args.novidium:
        tasks = NOVIDIUM_TASKS
    elif args.tasks:
        all_tasks = ABLATION_TASKS + NOVIDIUM_TASKS
        tasks = [t for t in all_tasks if t["id"] in args.tasks]
    else:
        tasks = ABLATION_TASKS

    run_ablation(args.api_url, tasks, direct=not args.api_mode,
                 include_smart=args.include_smart or args.smart_only)


if __name__ == "__main__":
    main()
