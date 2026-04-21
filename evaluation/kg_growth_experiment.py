#!/usr/bin/env python3
"""
Clean KG Growth Experiment — measures quality as KG grows from empty.

Protocol:
  1. Snapshot current KG (for safety)
  2. Wipe KG to empty
  3. Run tasks sequentially (KG writes enabled)
  4. After each batch, measure KG size, cumulative SR, physics, MPF
  5. Restore original KG

Usage inside container:
    python /app/evaluation/kg_growth_experiment.py --batch-size 10 --passes 2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# KG writes ENABLED for this experiment
os.environ.pop("KG_READ_ONLY", None)
os.environ["EXPERIMENT_PHASE"] = "kg_growth_clean"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation.benchmark_tasks_v2 import ALL_TASKS_V2

TASK_TIMEOUT = 180  # hard kill after 3 min per task


def _clear_run_nodes():
    """Delete all Run nodes and their relationships, keeping Material/Reference/schema."""
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pw = os.getenv("NEO4J_PASSWORD", "pde_neo4j_secret")
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        with driver.session() as s:
            before = s.run("MATCH (r:Run) RETURN count(r) AS c").single()["c"]
            s.run("MATCH (r:Run) DETACH DELETE r")
            after = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            print(f"  Deleted {before} Run nodes. {after} nodes remain (materials/refs).")
        driver.close()
    except Exception as e:
        print(f"  Warning: could not clear Run nodes: {e}")


@dataclass
class GrowthResult:
    task_idx: int
    batch_idx: int
    task_id: str
    difficulty: str
    success: bool
    wall_time_s: float
    run_id: str | None
    kg_node_count: int
    kg_rel_count: int
    physics_score: float
    property_fidelity: float
    cumulative_success_rate: float
    agent_iterations: int = 0
    error_message: str = ""


def _get_kg_stats() -> tuple[int, int]:
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        pw = os.getenv("NEO4J_PASSWORD", "pde_neo4j_secret")
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        with driver.session() as s:
            n = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        driver.close()
        return n, r
    except Exception:
        return -1, -1


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
    run_id = result.get("run_id")
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
    if run_id:
        for base in ("/workspace/results", "/app/results", "results"):
            cfg_path = Path(base) / run_id / "config.json"
            if cfg_path.exists():
                try:
                    with open(cfg_path) as f:
                        return json.load(f)
                except Exception:
                    pass
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
    checks = []
    for prop in ("k", "rho", "cp"):
        range_key = f"{prop}_range"
        if range_key in truth and prop in config:
            lo, hi = truth[range_key]
            val = config[prop]
            if lo <= val <= hi:
                checks.append(1.0)
            else:
                dist = min(abs(val - lo), abs(val - hi))
                span = hi - lo
                checks.append(max(0.0, 1.0 - dist / max(span, 1e-9)))
        elif prop in truth and prop in config:
            expected = truth[prop]
            actual = config[prop]
            if abs(expected) < 1e-12:
                checks.append(1.0 if abs(actual) < 1e-12 else 0.0)
            else:
                rel_err = abs(actual - expected) / abs(expected)
                checks.append(max(0.0, 1.0 - rel_err))
    return sum(checks) / max(len(checks), 1)


def _compute_physics_score(config: dict, truth: dict, sim_result: dict | None) -> float:
    mpf = _compute_property_fidelity(config, truth)
    t_score = 0.5
    if sim_result:
        t_checks = []
        for key in ("T_max_range", "T_min_range"):
            if key in truth:
                field_name = "max_temperature" if "max" in key else "min_temperature"
                actual = sim_result.get(field_name)
                if actual is not None:
                    lo, hi = truth[key]
                    if lo <= actual <= hi:
                        t_checks.append(1.0)
                    else:
                        dist = min(abs(actual - lo), abs(actual - hi))
                        span = max(hi - lo, 1.0)
                        t_checks.append(max(0.0, 1.0 - dist / span))
        if t_checks:
            t_score = sum(t_checks) / len(t_checks)
    return 0.5 * mpf + 0.5 * t_score


# ─── Subprocess-based task runner (hard-killable) ───────────────────────────

def _run_task_in_proc(description: str, result_queue: mp.Queue):
    """Run a single task inside a child process. Results go to queue."""
    try:
        from agents.simulation_agent import SimulationAgent
        agent = SimulationAgent(disable_kg=False, smart_kg=True)
        result = agent.run(description)
        result_queue.put(result)
    except Exception as exc:
        result_queue.put({"error": str(exc), "run_id": None,
                          "iterations": 0, "answer": ""})


def run_task_with_timeout(description: str, timeout: int = TASK_TIMEOUT) -> dict:
    """Run a task in a subprocess with a hard timeout.

    Uses multiprocessing so we can os.kill() if the child blocks in C code.
    """
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_run_task_in_proc, args=(description, q))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return {"error": f"Hard-killed after {timeout}s", "run_id": None,
                "iterations": 0, "answer": ""}

    if not q.empty():
        return q.get_nowait()
    return {"error": "Process exited without result", "run_id": None,
            "iterations": 0, "answer": ""}


def run_growth_experiment(
    batch_size: int = 10,
    seed: int = 42,
    passes: int = 2,
) -> dict:
    from kg_snapshot import save_snapshot, restore_snapshot

    # Build task list: `passes` shuffled copies of ALL_TASKS_V2
    tasks = []
    rng = random.Random(seed)
    for p in range(passes):
        copy = list(ALL_TASKS_V2)
        rng.shuffle(copy)
        for t in copy:
            tasks.append({**t, "pass": p + 1,
                          "id": f"{t['id']}_p{p+1}"})

    n_tasks = len(tasks)
    n_batches = (n_tasks + batch_size - 1) // batch_size

    print(f"{'='*70}")
    print(f"  KG GROWTH EXPERIMENT — Clean start")
    print(f"  Tasks: {n_tasks} ({passes} passes × {len(ALL_TASKS_V2)})")
    print(f"  Batch size: {batch_size}, Batches: {n_batches}")
    print(f"  Seed: {seed}, Timeout: {TASK_TIMEOUT}s/task")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")

    # Step 1: Save current KG
    print("Step 1: Saving current KG snapshot...")
    save_snapshot("pre_growth_experiment")

    # Step 2: Clear Run nodes (keep Material/Reference/static schema)
    print("Step 2: Clearing Run nodes (keeping material definitions)...")
    _clear_run_nodes()

    # Step 3: Warm up (in-process, single quick task)
    print("Step 3: Warming up...", end=" ", flush=True)
    try:
        from agents.simulation_agent import SimulationAgent
        warm_agent = SimulationAgent(disable_kg=False, smart_kg=True)
        warm_agent.run(
            "Run a minimal 2D heat equation: left T=0, right T=1, "
            "4x4 mesh, k=1, rho=1, cp=1, t_end=0.01, dt=0.005."
        )
        print("done.")
        del warm_agent
    except Exception as e:
        print(f"warmup note: {e}")

    # Step 4: Run tasks
    all_results: list[GrowthResult] = []
    total_success = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_tasks)
        batch = tasks[batch_start:batch_end]

        print(f"\n{'─'*70}")
        print(f"  BATCH {batch_idx+1}/{n_batches} "
              f"(tasks {batch_start+1}-{batch_end})")
        kg_n, kg_r = _get_kg_stats()
        print(f"  KG: {kg_n} nodes, {kg_r} rels")
        print(f"{'─'*70}")

        for i, task in enumerate(batch):
            global_idx = batch_start + i + 1
            print(f"  [{global_idx:3d}/{n_tasks}] {task['id']} "
                  f"({task['difficulty']}) ...", end=" ", flush=True)

            t0 = time.perf_counter()
            result = run_task_with_timeout(task["description"])
            wall = time.perf_counter() - t0

            run_id = result.get("run_id")
            iters = result.get("iterations", 0)
            error_msg = result.get("error", "")

            answer_str = str(result.get("answer", "")).lower()
            success = bool(run_id) and (
                "success" in answer_str and "fail" not in answer_str
            )
            if run_id and not success:
                sim_check = _load_sim_result(run_id)
                if sim_check and sim_check.get("status") == "success":
                    success = True

            config = _extract_config(result)

            total_success += int(success)
            cum_sr = total_success / global_idx

            sim_result = _load_sim_result(run_id)
            gt = task.get("ground_truth", {})
            mpf = _compute_property_fidelity(config, gt)
            phys = _compute_physics_score(config, gt, sim_result)

            kg_n_now, kg_r_now = _get_kg_stats()

            gr = GrowthResult(
                task_idx=global_idx,
                batch_idx=batch_idx,
                task_id=task["id"],
                difficulty=task["difficulty"],
                success=success,
                wall_time_s=wall,
                run_id=run_id,
                kg_node_count=kg_n_now,
                kg_rel_count=kg_r_now,
                physics_score=phys,
                property_fidelity=mpf,
                cumulative_success_rate=cum_sr,
                agent_iterations=iters,
                error_message=error_msg,
            )
            all_results.append(gr)

            status = "OK" if success else "FAIL"
            print(f"{status}  phys={phys:.2f}  mpf={mpf:.2f}  "
                  f"cum_sr={cum_sr:.2f}  kg={kg_n_now}n  "
                  f"t={wall:.1f}s  iter={iters}")

    # Step 5: Summary
    print(f"\n\n{'='*70}")
    print(f"  KG GROWTH EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total tasks: {n_tasks}")
    print(f"  Total success: {total_success} ({total_success/n_tasks*100:.1f}%)")
    final_kg_n, final_kg_r = _get_kg_stats()
    print(f"  Final KG: {final_kg_n} nodes, {final_kg_r} relationships")

    print(f"\n  {'Batch':<8s} {'Tasks':>6s} {'SR':>8s} {'Cum SR':>8s} "
          f"{'Phys':>8s} {'MPF':>8s} {'KG nodes':>10s}")
    for b in range(n_batches):
        br = [r for r in all_results if r.batch_idx == b]
        if not br:
            break
        b_sr = sum(1 for r in br if r.success) / len(br)
        cum = br[-1].cumulative_success_rate
        b_phys = sum(r.physics_score for r in br) / len(br)
        b_mpf = sum(r.property_fidelity for r in br) / len(br)
        kg_n = br[-1].kg_node_count
        print(f"  {b+1:<8d} {len(br):>6d} {b_sr:>8.0%} {cum:>8.2f} "
              f"{b_phys:>8.3f} {b_mpf:>8.3f} {kg_n:>10d}")

    # Save results
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "metadata": {
            "version": "kg_growth_v3",
            "n_tasks": n_tasks,
            "passes": passes,
            "batch_size": batch_size,
            "seed": seed,
            "timeout_per_task": TASK_TIMEOUT,
            "experiment_phase": "kg_growth_clean",
            "timestamp": datetime.now().isoformat(),
        },
        "results": [asdict(r) for r in all_results],
    }
    out_path = output_dir / "kg_growth_experiment.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # PGF data
    pgf_path = output_dir / "kg_growth.dat"
    with open(pgf_path, "w") as f:
        f.write("task_idx cum_sr kg_nodes phys mpf\n")
        for r in all_results:
            f.write(f"{r.task_idx} {r.cumulative_success_rate:.4f} "
                    f"{r.kg_node_count} {r.physics_score:.4f} "
                    f"{r.property_fidelity:.4f}\n")
    print(f"  PGF data saved: {pgf_path}")

    # Step 6: Restore original KG
    print("\nStep 6: Restoring original KG...")
    try:
        restore_snapshot("pre_growth_experiment")
        print("  Original KG restored.")
    except Exception as e:
        print(f"  WARNING: Could not restore KG: {e}")
        print(f"  Manual: python evaluation/kg_snapshot.py restore "
              f"--name pre_growth_experiment")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KG Growth Experiment")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, default=2,
                        help="Number of passes through the 50-task set")
    args = parser.parse_args()

    run_growth_experiment(
        batch_size=args.batch_size,
        seed=args.seed,
        passes=args.passes,
    )
