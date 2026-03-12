"""
Knowledge Graph Seeding Script
================================
Bypasses the LLM orchestrator and directly:
  1. Calls the FEniCSx runner API to solve each simulation
  2. Stores results in PostgreSQL via database operations
  3. Adds each run to the Neo4j knowledge graph

Run inside the agents container:
  docker exec -it pde-agents python3 /app/scripts/seed_knowledge_graph.py

Or from the host:
  docker exec pde-agents python3 /app/scripts/seed_knowledge_graph.py
"""

import sys, json, time, requests
sys.path.insert(0, "/app")

from database.operations import create_run, mark_run_finished
from database.models import RunStatus
from knowledge_graph.graph import get_kg

FENICS_URL = "http://fenics-runner:8080/run"

# ── 10 engineering materials with correct k/rho/cp matching Neo4j nodes ──────
MATERIALS = {
    "steel":        {"k": 50.0,  "rho": 7800.0, "cp": 500.0},
    "stainless316": {"k": 16.0,  "rho": 8000.0, "cp": 500.0},
    "aluminium":    {"k": 200.0, "rho": 2700.0, "cp": 900.0},
    "copper":       {"k": 385.0, "rho": 8960.0, "cp": 385.0},
    "titanium":     {"k": 6.7,   "rho": 4430.0, "cp": 526.0},
    "silicon":      {"k": 150.0, "rho": 2330.0, "cp": 700.0},
    "concrete":     {"k": 1.7,   "rho": 2300.0, "cp": 880.0},
    "glass":        {"k": 1.0,   "rho": 2230.0, "cp": 830.0},
    "water":        {"k": 0.6,   "rho": 1000.0, "cp": 4182.0},
    "air":          {"k": 0.026, "rho": 1.2,    "cp": 1005.0},
}

# ── Simulation plan ────────────────────────────────────────────────────────────
# Each entry: (run_id_suffix, dim, nx, ny, nz, bcs, t_end, dt, source, label)
def build_plan():
    plans = []

    # 1. All 10 materials — 2D Dirichlet (left hot, right cold)
    for mat, props in MATERIALS.items():
        # Choose dt for stability: small dt for high diffusivity materials
        alpha = props["k"] / (props["rho"] * props["cp"])
        # Characteristic time ~ L²/alpha; use ~100 time steps to reach SS
        t_end = min(max(5 * (0.04**2 / alpha), 1.0), 5000.0)
        dt = t_end / 100
        dt = max(dt, 0.01)

        plans.append({
            "run_id": f"seed_{mat}_2d",
            "dim": 2, "nx": 40, "ny": 20, "nz": None,
            "t_end": round(t_end, 2), "dt": round(dt, 4),
            "source": 0.0, "theta": 1.0,
            "u_init": 300.0,
            "bcs": [
                {"type": "dirichlet", "value": 800.0, "location": "left"},
                {"type": "dirichlet", "value": 300.0, "location": "right"},
                {"type": "neumann",   "value": 0.0,   "location": "top"},
                {"type": "neumann",   "value": 0.0,   "location": "bottom"},
            ],
            **props,
        })

    # 2. Steel — varying sizes (aspect ratio study)
    for (nx, ny, label) in [(20, 20, "sq"), (80, 20, "wide"), (20, 80, "tall")]:
        plans.append({
            "run_id": f"seed_steel_2d_{label}",
            "dim": 2, "nx": nx, "ny": ny, "nz": None,
            "t_end": 20.0, "dt": 0.5, "source": 0.0, "theta": 1.0,
            "u_init": 300.0,
            "bcs": [
                {"type": "dirichlet", "value": 800.0, "location": "left"},
                {"type": "dirichlet", "value": 300.0, "location": "right"},
                {"type": "neumann",   "value": 0.0,   "location": "top"},
                {"type": "neumann",   "value": 0.0,   "location": "bottom"},
            ],
            **MATERIALS["steel"],
        })

    # 3. Robin (convective) boundary conditions — aluminium and steel
    for mat in ["aluminium", "steel"]:
        props = MATERIALS[mat]
        alpha = props["k"] / (props["rho"] * props["cp"])
        t_end = min(max(3 * (0.04**2 / alpha), 1.0), 1000.0)
        dt = max(t_end / 80, 0.01)
        plans.append({
            "run_id": f"seed_{mat}_robin",
            "dim": 2, "nx": 32, "ny": 32, "nz": None,
            "t_end": round(t_end, 2), "dt": round(dt, 4),
            "source": 0.0, "theta": 1.0,
            "u_init": 293.0,
            "bcs": [
                {"type": "dirichlet", "value": 800.0, "location": "left"},
                {"type": "dirichlet", "value": 300.0, "location": "right"},
                {"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": "top"},
                {"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": "bottom"},
            ],
            **props,
        })

    # 4. Internal heat source — concrete and silicon
    for mat in ["concrete", "silicon"]:
        props = MATERIALS[mat]
        plans.append({
            "run_id": f"seed_{mat}_source",
            "dim": 2, "nx": 32, "ny": 32, "nz": None,
            "t_end": 500.0, "dt": 5.0, "source": 5000.0, "theta": 1.0,
            "u_init": 293.0,
            "bcs": [
                {"type": "dirichlet", "value": 293.0, "location": "left"},
                {"type": "dirichlet", "value": 293.0, "location": "right"},
                {"type": "neumann",   "value": 0.0,   "location": "top"},
                {"type": "neumann",   "value": 0.0,   "location": "bottom"},
            ],
            **props,
        })

    # 5. 3D runs — steel, aluminium, copper
    for mat in ["steel", "aluminium", "copper"]:
        props = MATERIALS[mat]
        alpha = props["k"] / (props["rho"] * props["cp"])
        t_end = min(max(2 * (0.04**2 / alpha), 1.0), 500.0)
        dt = max(t_end / 60, 0.05)
        plans.append({
            "run_id": f"seed_{mat}_3d",
            "dim": 3, "nx": 16, "ny": 16, "nz": 16,
            "t_end": round(t_end, 2), "dt": round(dt, 4),
            "source": 0.0, "theta": 1.0,
            "u_init": 300.0,
            "bcs": [
                {"type": "dirichlet", "value": 800.0, "location": "left"},
                {"type": "dirichlet", "value": 300.0, "location": "right"},
                {"type": "neumann",   "value": 0.0,   "location": "top"},
                {"type": "neumann",   "value": 0.0,   "location": "bottom"},
                {"type": "neumann",   "value": 0.0,   "location": "front"},
                {"type": "neumann",   "value": 0.0,   "location": "back"},
            ],
            **props,
        })

    # 6. Deliberate rule violations — creates TRIGGERED edges in the KG
    # INCONSISTENT_IC: u_init far from BCs
    plans.append({
        "run_id": "seed_rule_inconsistent_ic",
        "dim": 2, "nx": 32, "ny": 32, "nz": None,
        "t_end": 10.0, "dt": 0.5, "source": 0.0, "theta": 1.0,
        "u_init": 0.0,   # <-- far from BCs at 300/800 K
        "bcs": [
            {"type": "dirichlet", "value": 800.0, "location": "left"},
            {"type": "dirichlet", "value": 300.0, "location": "right"},
            {"type": "neumann",   "value": 0.0,   "location": "top"},
            {"type": "neumann",   "value": 0.0,   "location": "bottom"},
        ],
        **MATERIALS["steel"],
    })
    # SHORT_SIMULATION: t_end < 5*dt
    plans.append({
        "run_id": "seed_rule_short_sim",
        "dim": 2, "nx": 32, "ny": 32, "nz": None,
        "t_end": 0.3, "dt": 0.1, "source": 0.0, "theta": 1.0,
        "u_init": 300.0,
        "bcs": [
            {"type": "dirichlet", "value": 800.0, "location": "left"},
            {"type": "dirichlet", "value": 300.0, "location": "right"},
            {"type": "neumann",   "value": 0.0,   "location": "top"},
            {"type": "neumann",   "value": 0.0,   "location": "bottom"},
        ],
        **MATERIALS["copper"],
    })
    # LARGE_DT_RELATIVE_TO_DIFFUSION: dt >> h²/alpha
    plans.append({
        "run_id": "seed_rule_large_dt",
        "dim": 2, "nx": 8, "ny": 8, "nz": None,
        "t_end": 100.0, "dt": 50.0, "source": 0.0, "theta": 1.0,
        "u_init": 300.0,
        "bcs": [
            {"type": "dirichlet", "value": 500.0, "location": "left"},
            {"type": "dirichlet", "value": 300.0, "location": "right"},
            {"type": "neumann",   "value": 0.0,   "location": "top"},
            {"type": "neumann",   "value": 0.0,   "location": "bottom"},
        ],
        **MATERIALS["aluminium"],
    })

    return plans


def run_and_store(plan: dict, kg, dry_run: bool = False) -> bool:
    run_id = plan["run_id"]
    print(f"\n  {'[DRY]' if dry_run else ''} {run_id} ...", flush=True)

    payload = {
        "run_id":     run_id,
        "dim":        plan["dim"],
        "nx":         plan["nx"],
        "ny":         plan["ny"],
        "k":          plan["k"],
        "rho":        plan["rho"],
        "cp":         plan["cp"],
        "source":     plan.get("source", 0.0),
        "u_init":     plan["u_init"],
        "bcs":        plan["bcs"],
        "t_end":      plan["t_end"],
        "dt":         plan["dt"],
        "theta":      plan.get("theta", 1.0),
        "output_dir": "/workspace/results",
    }
    if plan["dim"] == 3:
        payload["nz"] = plan["nz"]

    if dry_run:
        print(f"    Would POST: dim={plan['dim']}, k={plan['k']}, "
              f"t_end={plan['t_end']}, dt={plan['dt']}")
        return True

    # 1. Create DB record (no db session arg — operations.py manages its own)
    try:
        create_run(run_id, plan)
    except Exception as e:
        # Likely already exists — proceed anyway
        print(f"    create_run warning: {e}")

    # 2. Run FEniCSx
    t0 = time.time()
    try:
        resp = requests.post(FENICS_URL, json=payload, timeout=300)
        result = resp.json()
    except Exception as e:
        print(f"    ERROR calling FEniCSx: {e}")
        try:
            mark_run_finished(run_id, {}, RunStatus.FAILED)
        except Exception:
            pass
        return False

    elapsed = time.time() - t0

    if result.get("status") != "success":
        err = result.get("error", "unknown")
        print(f"    FAILED ({elapsed:.1f}s): {err}")
        try:
            mark_run_finished(run_id, result, RunStatus.FAILED)
        except Exception:
            pass
        return False

    # 3. Mark as success in DB
    try:
        mark_run_finished(run_id, result, RunStatus.SUCCESS)
    except Exception as e:
        print(f"    mark_run_finished warning: {e}")

    # 4. Add to knowledge graph
    warnings = []
    try:
        from knowledge_graph.rules import check_config as check_config_warnings
        warnings = check_config_warnings(plan)
    except Exception:
        pass

    try:
        kg.add_run(run_id, plan, result, [w["code"] for w in warnings])
        kg_ok = "✓ KG"
    except Exception as e:
        kg_ok = f"✗ KG({e})"

    t_max = result.get("t_max", "?")
    t_min = result.get("t_min", "?")
    wall  = result.get("wall_time", elapsed)
    t_str = f"T=[{float(t_min):.0f}..{float(t_max):.0f}]K" if t_min != "?" else "T=?"
    w_str = f"  warnings={[w['code'] for w in warnings]}" if warnings else ""
    print(f"    ✓ done {elapsed:.1f}s  {t_str}  "
          f"wall={float(wall):.2f}s  {kg_ok}{w_str}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running simulations")
    parser.add_argument("--filter", default="",
                        help="Only run IDs containing this string (e.g. steel, 3d, rule)")
    args = parser.parse_args()

    plans = build_plan()
    if args.filter:
        plans = [p for p in plans if args.filter in p["run_id"]]

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Knowledge Graph Seeding")
    print(f"{'=' * 60}")
    print(f"  Total simulations planned: {len(plans)}")
    mat_coverage = set(p["run_id"].split("_")[1] for p in plans if "seed_" in p["run_id"])
    print(f"  Materials covered: {sorted(mat_coverage)}")
    print(f"  Rule violations included: "
          f"{sum(1 for p in plans if 'rule' in p['run_id'])}")
    print()

    if not args.dry_run:
        kg = get_kg()
    else:
        kg = None

    ok = fail = skip = 0
    for plan in plans:
        success = run_and_store(plan, kg, dry_run=args.dry_run)
        if args.dry_run:
            skip += 1
        elif success:
            ok += 1
        else:
            fail += 1

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"  Dry run complete — {skip} simulations would be run")
    else:
        print(f"  Done: {ok} succeeded, {fail} failed")
        if kg:
            stats = kg.stats()
            print(f"  KG now has: {stats}")


if __name__ == "__main__":
    main()
