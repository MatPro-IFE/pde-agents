#!/usr/bin/env python3
"""
Ablation study v2 — methodologically sound.

Key differences from v1:
  1. KG is FROZEN (KG_READ_ONLY=true) — no writes during ablation
  2. 50 tasks per mode (vs 7-10)
  3. Tasks are shuffled independently per mode to avoid ordering bias
  4. All runs tagged with EXPERIMENT_PHASE for traceability
  5. Each mode uses a freshly created agent (no warm-up carryover)
  6. Statistical analysis with CIs built in

Usage inside the container:
    # Snapshot the KG first
    python /app/evaluation/kg_snapshot.py save --name pre_ablation_v2

    # Run the ablation (KG frozen automatically)
    python /app/evaluation/ablation/run_ablation_v2.py --direct

    # Or run a specific mode only
    python /app/evaluation/ablation/run_ablation_v2.py --direct --modes kg_off kg_smart
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path


class TaskTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TaskTimeout("Task exceeded time limit")

# Enforce KG read-only BEFORE any other imports that might load tools
os.environ["KG_READ_ONLY"] = "true"
os.environ["EXPERIMENT_PHASE"] = "ablation_v2"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ablation.benchmark_tasks_v2 import ALL_TASKS_V2, get_tasks_by_difficulty

TIMEOUT = 420  # 7 minutes per task (allows for agent-internal retry)
MAX_RETRIES = 1  # Agent handles retries internally via _AUTO_RETRY


@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    mode: str
    success: bool
    wall_time_s: float
    agent_iterations: int
    run_id: str | None
    config_produced: dict
    config_quality_score: float
    first_try_success: bool
    error_message: str = ""
    raw_response: dict = field(default_factory=dict)
    property_fidelity: float = 0.0
    t_max_actual: float | None = None
    t_min_actual: float | None = None
    t_mean_actual: float | None = None
    physics_score: float = 0.0


def _load_sim_result(run_id: str) -> dict | None:
    if not run_id:
        return None
    for base in ("/workspace/results", "/app/results", "results"):
        p = Path(base) / run_id / "result.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def _extract_config(result: dict) -> dict:
    """Extract the simulation config the agent used.

    Priority: 1) Database (config_json), 2) tool_calls_log, 3) answer text.
    """
    run_id = result.get("run_id")

    # Strategy 1: Read config from the database
    if run_id:
        try:
            from database.operations import get_session_factory
            from database.models import SimulationRun
            from sqlalchemy import select
            sf = get_session_factory()
            with sf() as session:
                run = session.execute(
                    select(SimulationRun).where(SimulationRun.run_id == run_id)
                ).scalar_one_or_none()
                if run and run.config_json and isinstance(run.config_json, dict):
                    return run.config_json
        except Exception:
            pass

    # Strategy 2: Read config from the result.json on disk
    if run_id:
        for base in ("/workspace/results", "/app/results", "results"):
            cfg_path = Path(base) / run_id / "config.json"
            if cfg_path.exists():
                try:
                    with open(cfg_path) as f:
                        return json.load(f)
                except Exception:
                    pass

    # Strategy 3: Mine tool_calls_log for run_simulation arguments
    for entry in result.get("tool_calls_log", []):
        if isinstance(entry, dict):
            tool_name = entry.get("tool_name", entry.get("name", ""))
            if tool_name in ("run_simulation", "validate_config"):
                args = entry.get("args", entry.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if isinstance(args, dict):
                    cfg = args.get("config_json", args.get("config", args))
                    if isinstance(cfg, str):
                        try:
                            cfg = json.loads(cfg)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    if isinstance(cfg, dict) and any(k in cfg for k in ("nx", "k", "dim")):
                        return cfg

    return {}


def _compute_property_fidelity(config: dict, truth: dict) -> float:
    """Measure how close k/rho/cp are to ground truth. 1.0 = perfect match."""
    errors = []
    for prop in ("k", "rho", "cp"):
        range_key = f"{prop}_range"
        if range_key in truth:
            lo, hi = truth[range_key]
            midpoint = (lo + hi) / 2.0
            val = config.get(prop)
            if val is None:
                errors.append(1.0)
            else:
                rel_err = abs(val - midpoint) / max(midpoint, 1e-9)
                errors.append(min(rel_err, 1.0))
        elif prop in truth:
            val = config.get(prop)
            if val is not None and abs(val - truth[prop]) < 0.01 * abs(truth[prop] + 1e-9):
                errors.append(0.0)
            else:
                errors.append(1.0)
    if not errors:
        return 1.0
    return max(0.0, 1.0 - sum(errors) / len(errors))


def _compute_physics_score(config: dict, truth: dict, sim_result: dict | None) -> float:
    """Combined score: 0.5 * property_fidelity + 0.5 * temperature_score."""
    mpf = _compute_property_fidelity(config, truth)

    temp_score = 1.0
    if sim_result:
        for key, result_key in [("T_max_range", "max_temperature"),
                                 ("T_min_range", "min_temperature")]:
            if key in truth:
                actual = sim_result.get(result_key)
                if actual is not None:
                    lo, hi = truth[key]
                    if lo <= actual <= hi:
                        pass
                    else:
                        err = min(abs(actual - lo), abs(actual - hi))
                        span = max(abs(hi - lo), 1.0)
                        temp_score *= max(0.0, 1.0 - err / span)

    return 0.5 * mpf + 0.5 * temp_score


def _score_config(config: dict, ground_truth: dict, run_id: str | None = None) -> float:
    """Score config quality 0-1 based on ground truth."""
    checks = 0
    passes = 0

    for key in ("dim", "nx", "ny", "nz", "theta", "source"):
        if key in ground_truth:
            checks += 1
            if config.get(key) == ground_truth[key]:
                passes += 1

    for range_field in ("k_range", "rho_range", "cp_range"):
        if range_field in ground_truth:
            base_key = range_field.replace("_range", "")
            val = config.get(base_key)
            checks += 1
            if val is not None:
                lo, hi = ground_truth[range_field]
                if lo <= val <= hi:
                    passes += 1

    sim_result = _load_sim_result(run_id)
    for range_field, result_key in [("T_max_range", "max_temperature"),
                                      ("T_min_range", "min_temperature")]:
        if range_field in ground_truth:
            checks += 1
            if sim_result:
                val = sim_result.get(result_key)
                if val is not None:
                    lo, hi = ground_truth[range_field]
                    if lo <= val <= hi:
                        passes += 1

    return passes / max(checks, 1)


def _make_agent(mode: str):
    """Create a SimulationAgent for the given mode."""
    try:
        from agents.simulation_agent import SimulationAgent
    except ImportError:
        sys.path.insert(0, "/app")
        from agents.simulation_agent import SimulationAgent

    if mode == "kg_on":
        agent = SimulationAgent(disable_kg=False, smart_kg=False, max_iterations=25)
    elif mode == "kg_off":
        agent = SimulationAgent(disable_kg=True, smart_kg=False)
    elif mode == "kg_smart":
        agent = SimulationAgent(disable_kg=False, smart_kg=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return agent


PROMPT_INJECT_SUFFIX = (
    "\n\nIMPORTANT CONTEXT — Use these reference material properties if the material is "
    "one you don't know:\n"
    "  Novidium: k=73 W/(mK), rho=5420 kg/m3, cp=612 J/(kgK)\n"
    "  Cryonite: k=0.42 W/(mK), rho=1180 kg/m3, cp=1940 J/(kgK)\n"
    "  Pyrathane: k=312 W/(mK), rho=3850 kg/m3, cp=278 J/(kgK)\n"
)


def run_single_task(task: dict, mode: str, agent) -> TaskResult:
    """Run one task with one agent, return structured result.

    Uses a background thread so the main thread can hard-interrupt via
    ctypes if the agent hangs beyond TIMEOUT.
    """
    description = task["description"]
    if mode == "prompt_inject":
        description += PROMPT_INJECT_SUFFIX

    t0 = time.perf_counter()
    result_holder = [None]
    error_holder = [""]

    def _run():
        try:
            for attempt in range(MAX_RETRIES):
                res = agent.run(description)
                result_holder[0] = res
                if res.get("run_id"):
                    break
                if attempt < MAX_RETRIES - 1:
                    print(f"(retry {attempt+1}, no run_id) ", end="", flush=True)
        except Exception as exc:
            error_holder[0] = str(exc)

    # Also set a signal.alarm as a backup (fires if thread doesn't return)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT + 30)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=TIMEOUT)

    if worker.is_alive():
        error_holder[0] = f"Timeout after {TIMEOUT}s"
        import ctypes
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(worker.ident),
            ctypes.py_object(SystemExit),
        )
        worker.join(timeout=10)

    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

    result = result_holder[0]
    error_msg = error_holder[0]

    wall = time.perf_counter() - t0

    if result is None:
        return TaskResult(
            task_id=task["id"], difficulty=task["difficulty"], mode=mode,
            success=False, wall_time_s=wall, agent_iterations=0,
            run_id=None, config_produced={}, config_quality_score=0.0,
            first_try_success=False, error_message=error_msg or "No result",
        )

    run_id = result.get("run_id")
    answer = result.get("answer", "")
    iters = result.get("iterations", 1)

    # Success = got a run_id AND answer mentions "success" (not "fail")
    answer_str = str(answer).lower()
    success = bool(run_id) and (
        "success" in answer_str and "fail" not in answer_str
    )
    # Also count as success if run_id exists and sim result exists
    if run_id and not success:
        sim_check = _load_sim_result(run_id)
        if sim_check and sim_check.get("status") == "success":
            success = True

    # Extract config from DB / disk / tool calls
    config = _extract_config(result)

    first_try = iters <= 12 and success
    quality = _score_config(config, task.get("ground_truth", {}), run_id=run_id)

    sim_result = _load_sim_result(run_id)
    mpf = _compute_property_fidelity(config, task.get("ground_truth", {}))
    phys = _compute_physics_score(config, task.get("ground_truth", {}), sim_result)

    return TaskResult(
        task_id=task["id"], difficulty=task["difficulty"], mode=mode,
        success=success, wall_time_s=wall, agent_iterations=iters,
        run_id=run_id, config_produced=config, config_quality_score=quality,
        first_try_success=first_try, error_message=result.get("error", ""),
        property_fidelity=mpf, physics_score=phys,
        t_max_actual=sim_result.get("max_temperature") if sim_result else None,
        t_min_actual=sim_result.get("min_temperature") if sim_result else None,
        t_mean_actual=sim_result.get("mean_temperature") if sim_result else None,
    )


def aggregate(results: list[TaskResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    successes = [r for r in results if r.success]
    first_tries = [r for r in results if r.first_try_success]
    sr = len(successes) / n
    return {
        "n": n,
        "success_rate": sr,
        "first_try_rate": len(first_tries) / n,
        "avg_quality": sum(r.config_quality_score for r in results) / n,
        "avg_property_fidelity": sum(r.property_fidelity for r in results) / n,
        "avg_physics_score": sum(r.physics_score for r in results) / n,
        "avg_iterations": sum(r.agent_iterations for r in results) / n,
        "avg_wall_time": sum(r.wall_time_s for r in results) / n,
        "success_ci_95": 1.96 * (sr * (1 - sr) / n) ** 0.5 if n > 1 else 0,
    }


def aggregate_by_difficulty(results: list[TaskResult]) -> dict:
    by_diff = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r)
    return {d: aggregate(rs) for d, rs in sorted(by_diff.items())}


def _incremental_save(mode, results, tasks, seed, merge_path):
    """Save partial results to disk so progress isn't lost on crash."""
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "ablation_v2_results.json"

    partial = {
        "metadata": {
            "version": "v2",
            "n_tasks": len(tasks),
            "seed": seed,
            "kg_read_only": True,
            "experiment_phase": "ablation_v2",
            "last_updated": datetime.now().isoformat(),
            "partial": True,
        },
        mode: {
            "aggregate": aggregate(results),
            "by_difficulty": aggregate_by_difficulty(results),
            "tasks": [asdict(r) for r in results],
        },
    }

    if merge_path and Path(merge_path).exists():
        try:
            with open(merge_path) as f:
                existing = json.load(f)
            existing[mode] = partial[mode]
            existing["metadata"]["last_updated"] = datetime.now().isoformat()
            partial = existing
        except (json.JSONDecodeError, KeyError):
            pass

    with open(out_path, "w") as f:
        json.dump(partial, f, indent=2, default=str)


def run_ablation_v2(
    modes: list[str] | None = None,
    tasks: list[dict] | None = None,
    seed: int = 42,
    merge_path: str | None = None,
) -> dict:
    """Run the full ablation study with frozen KG.

    Args:
        modes: Which modes to run. Default: all three.
        tasks: Task list. Default: ALL_TASKS_V2 (50 tasks).
        seed: Random seed for task shuffling.
        merge_path: Path to existing results JSON to merge new modes into.
    """
    if modes is None:
        modes = ["kg_on", "kg_off", "kg_smart"]
    if tasks is None:
        tasks = ALL_TASKS_V2

    print(f"{'='*70}")
    print(f"  ABLATION STUDY v2 — Frozen KG, {len(tasks)} tasks × {len(modes)} modes")
    print(f"  KG_READ_ONLY={os.environ.get('KG_READ_ONLY')}")
    print(f"  EXPERIMENT_PHASE={os.environ.get('EXPERIMENT_PHASE')}")
    print(f"  Seed={seed}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    all_results = {}
    rng = random.Random(seed)

    for mode in modes:
        print(f"\n{'─'*70}")
        print(f"  MODE: {mode.upper()} ({len(tasks)} tasks)")
        print(f"{'─'*70}")

        # Shuffle tasks independently per mode
        shuffled = list(tasks)
        rng.shuffle(shuffled)

        # Create a fresh agent per mode
        print(f"  Creating {mode} agent...", end=" ", flush=True)
        agent = _make_agent(mode)
        print("done.")

        # Warm up with a trivial task (not counted) — skip for KG On
        # to avoid mandatory KG query overhead on the warmup task
        if mode != "kg_on":
            print("  Warming up...", end=" ", flush=True)
            saved_retry = getattr(agent, '_AUTO_RETRY', 1)
            agent._AUTO_RETRY = 1  # no retry during warmup
            try:
                agent.run(
                    "Run a minimal 2D heat equation: left T=0, right T=1, "
                    "4x4 mesh, k=1, rho=1, cp=1, t_end=0.01, dt=0.005."
                )
                print("done.")
            except Exception as e:
                print(f"warmup failed: {e}")
            finally:
                agent._AUTO_RETRY = saved_retry
        else:
            print("  (skipping warmup for KG On)")

        results = []
        for i, task in enumerate(shuffled):
            print(f"  [{i+1:3d}/{len(shuffled)}] {task['id']} ({task['difficulty']}) ...",
                  end=" ", flush=True)
            tr = run_single_task(task, mode, agent)
            results.append(tr)
            status = "OK" if tr.success else "FAIL"
            print(f"{status}  phys={tr.physics_score:.2f}  mpf={tr.property_fidelity:.2f}  "
                  f"t={tr.wall_time_s:.1f}s  iter={tr.agent_iterations}")

            # Incremental save every 5 tasks to avoid losing progress
            if (i + 1) % 5 == 0 or (i + 1) == len(shuffled):
                _incremental_save(mode, results, tasks, seed, merge_path)

        all_results[mode] = results

    # Print summary
    print(f"\n\n{'='*70}")
    print(f"  ABLATION v2 RESULTS SUMMARY ({len(tasks)} tasks/mode)")
    print(f"{'='*70}")
    header = f"  {'Metric':<28s}"
    for mode in modes:
        header += f"  {mode:>12s}"
    print(header)
    print("  " + "-" * (28 + 14 * len(modes)))

    aggs = {mode: aggregate(results) for mode, results in all_results.items()}
    for metric in ("success_rate", "success_ci_95", "first_try_rate",
                    "avg_quality", "avg_property_fidelity", "avg_physics_score",
                    "avg_iterations", "avg_wall_time"):
        row = f"  {metric:<28s}"
        for mode in modes:
            v = aggs[mode].get(metric, 0)
            fmt = ".3f" if "rate" in metric or "quality" in metric or "fidelity" in metric or "score" in metric or "ci" in metric else ".1f"
            row += f"  {v:>12{fmt}}"
        print(row)

    # Per-difficulty breakdown
    print(f"\n  {'─'*60}")
    print(f"  PER-DIFFICULTY BREAKDOWN")
    for mode in modes:
        print(f"\n  {mode.upper()}:")
        by_d = aggregate_by_difficulty(all_results[mode])
        for d, a in by_d.items():
            print(f"    {d:<8s}: SR={a['success_rate']:.2f} ±{a['success_ci_95']:.2f}  "
                  f"phys={a['avg_physics_score']:.2f}  mpf={a['avg_property_fidelity']:.2f}  "
                  f"n={a['n']}")

    # Save results
    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "metadata": {
            "version": "v2",
            "n_tasks": len(tasks),
            "modes": modes,
            "seed": seed,
            "kg_read_only": True,
            "experiment_phase": "ablation_v2",
            "timestamp": datetime.now().isoformat(),
        },
    }
    for mode in modes:
        summary[mode] = {
            "aggregate": aggs[mode],
            "by_difficulty": aggregate_by_difficulty(all_results[mode]),
            "tasks": [asdict(r) for r in all_results[mode]],
        }

    out_path = output_dir / "ablation_v2_results.json"

    # If --merge, load existing results and replace only the modes we just ran
    if merge_path and Path(merge_path).exists():
        with open(merge_path) as f:
            existing = json.load(f)
        for mode in modes:
            existing[mode] = summary[mode]
        existing["metadata"]["modes"] = list(existing.keys() - {"metadata"})
        existing["metadata"]["last_updated"] = datetime.now().isoformat()
        summary = existing
        print(f"\n  Merged into existing results ({merge_path})")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study v2")
    parser.add_argument("--modes", nargs="+",
                        choices=["kg_on", "kg_off", "kg_smart"],
                        help="Which modes to run (default: all three)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct", action="store_true",
                        help="Run agents directly (default inside container)")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter tasks by difficulty level")
    parser.add_argument("--merge", type=str, default=None,
                        help="Path to existing results JSON to merge into")
    args = parser.parse_args()

    tasks = get_tasks_by_difficulty(args.difficulty)
    run_ablation_v2(modes=args.modes, tasks=tasks, seed=args.seed,
                    merge_path=args.merge)
