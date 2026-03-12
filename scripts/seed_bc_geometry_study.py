"""
BC × Material × Geometry Knowledge Graph Study
================================================
Systematically builds knowledge across three axes:

  Axis 1 — Boundary condition type
    • dirichlet_only    : fixed T both sides, insulated top/bottom
    • one_sided_flux    : fixed T left, insulated right (semi-infinite analog)
    • neumann_flux_out  : fixed T left, prescribed heat flux right
    • dirichlet_robin   : fixed T walls, convective cooling top/bottom
    • full_robin        : convective on all faces (electronics cooling)
    • heat_source_cool  : internal heat source, cooled walls
    • heat_source_robin : internal heat source, convective cooling all faces

  Axis 2 — Material (all 10 seeded materials)
    copper, aluminium, silicon, steel, stainless316,
    titanium, concrete, glass, water, air

  Axis 3 — Geometry (physical domain size in metres)
    tiny   : 0.01 m × 0.01 m  (10 mm — MEMS / microelectronics)
    small  : 0.04 m × 0.02 m  (40 mm × 20 mm — component)
    medium : 0.10 m × 0.05 m  (10 cm × 5 cm  — panel)
    large  : 0.30 m × 0.10 m  (30 cm × 10 cm — structural element)

The script skips already-existing run_ids so it is safe to re-run incrementally.

Run inside the agents container:
    docker exec pde-agents python3 /app/scripts/seed_bc_geometry_study.py --dry-run
    docker exec -d pde-agents bash -c "python3 /app/scripts/seed_bc_geometry_study.py > /tmp/bc_geo.log 2>&1"
    docker exec pde-agents tail -f /tmp/bc_geo.log
"""

import sys, json, time, requests, argparse
sys.path.insert(0, "/app")

from database.operations import create_run, mark_run_finished
from database.models import RunStatus
from knowledge_graph.graph import get_kg
from knowledge_graph.rules import check_config as check_config_warnings

FENICS_URL = "http://fenics-runner:8080/run"

# ── Materials ─────────────────────────────────────────────────────────────────
MATERIALS = {
    "copper":       {"k": 385.0, "rho": 8960.0, "cp": 385.0},
    "aluminium":    {"k": 200.0, "rho": 2700.0, "cp": 900.0},
    "silicon":      {"k": 150.0, "rho": 2330.0, "cp": 700.0},
    "steel":        {"k": 50.0,  "rho": 7800.0, "cp": 500.0},
    "stainless316": {"k": 16.0,  "rho": 8000.0, "cp": 500.0},
    "titanium":     {"k": 6.7,   "rho": 4430.0, "cp": 526.0},
    "concrete":     {"k": 1.7,   "rho": 2300.0, "cp": 880.0},
    "glass":        {"k": 1.0,   "rho": 2230.0, "cp": 830.0},
    "water":        {"k": 0.6,   "rho": 1000.0, "cp": 4182.0},
    "air":          {"k": 0.026, "rho": 1.2,    "cp": 1005.0},
}

# ── Geometries ────────────────────────────────────────────────────────────────
# (label, Lx, Ly)  — mesh density scales with size so h stays ~constant
GEOMETRIES = [
    ("tiny",   0.01, 0.01, 16, 16),   # h ~ 0.6 mm
    ("small",  0.04, 0.02, 40, 20),   # h ~ 1 mm
    ("medium", 0.10, 0.05, 50, 25),   # h ~ 2 mm
    ("large",  0.30, 0.10, 60, 20),   # h ~ 5 mm
]

# ── BC configurations ─────────────────────────────────────────────────────────
def make_bcs(bc_type: str, T_hot=800.0, T_cold=300.0,
             h_conv=25.0, T_amb=293.0, q_flux=5000.0) -> list:
    """Return a list of BC dicts for the named configuration."""
    insulated = {"type": "neumann", "value": 0.0}

    configs = {
        # Fixed T both sides; top/bottom perfectly insulated
        "dirichlet_only": [
            {"type": "dirichlet", "value": T_hot,  "location": "left"},
            {"type": "dirichlet", "value": T_cold, "location": "right"},
            {**insulated, "location": "top"},
            {**insulated, "location": "bottom"},
        ],
        # Fixed T left only; right is insulated (models semi-infinite slab)
        "one_sided_flux": [
            {"type": "dirichlet", "value": T_hot, "location": "left"},
            {**insulated, "location": "right"},
            {**insulated, "location": "top"},
            {**insulated, "location": "bottom"},
        ],
        # Fixed T left; prescribed outward heat flux on right
        "neumann_flux_out": [
            {"type": "dirichlet", "value": T_hot, "location": "left"},
            {"type": "neumann",   "value": q_flux, "location": "right"},
            {**insulated, "location": "top"},
            {**insulated, "location": "bottom"},
        ],
        # Fixed T walls; convective cooling on top and bottom (fin/channel)
        "dirichlet_robin": [
            {"type": "dirichlet", "value": T_hot,  "location": "left"},
            {"type": "dirichlet", "value": T_cold, "location": "right"},
            {"type": "robin", "alpha": h_conv, "u_inf": T_amb, "location": "top"},
            {"type": "robin", "alpha": h_conv, "u_inf": T_amb, "location": "bottom"},
        ],
        # Convective cooling on all faces (component in forced-air flow)
        "full_robin": [
            {"type": "robin", "alpha": h_conv, "u_inf": T_amb, "location": "left"},
            {"type": "robin", "alpha": h_conv, "u_inf": T_amb, "location": "right"},
            {"type": "robin", "alpha": h_conv, "u_inf": T_amb, "location": "top"},
            {"type": "robin", "alpha": h_conv, "u_inf": T_amb, "location": "bottom"},
        ],
        # Internal heat source with fixed coolant walls
        "heat_source_cool": [
            {"type": "dirichlet", "value": T_cold, "location": "left"},
            {"type": "dirichlet", "value": T_cold, "location": "right"},
            {**insulated, "location": "top"},
            {**insulated, "location": "bottom"},
        ],
        # Internal heat source with convective cooling (PCB / chip cooling)
        "heat_source_robin": [
            {"type": "robin", "alpha": h_conv * 4, "u_inf": T_amb, "location": "left"},
            {"type": "robin", "alpha": h_conv * 4, "u_inf": T_amb, "location": "right"},
            {"type": "robin", "alpha": h_conv * 4, "u_inf": T_amb, "location": "top"},
            {"type": "robin", "alpha": h_conv * 4, "u_inf": T_amb, "location": "bottom"},
        ],
    }
    return configs[bc_type]


BC_SOURCES = {
    "heat_source_cool":  10_000.0,   # W/m³
    "heat_source_robin": 20_000.0,
}

BC_UINIT = {
    "full_robin":        293.0,
    "heat_source_cool":  300.0,
    "heat_source_robin": 293.0,
}


def stable_time_params(k, rho, cp, Lx, Ly, nx, ny, t_multiplier=3.0, max_t=10_000.0):
    """Return (t_end, dt) scaled to the physical domain and material diffusivity."""
    alpha = k / (rho * cp)
    # Characteristic length = shorter dimension
    L_char = min(Lx, Ly)
    # Characteristic time to reach ~95% of steady state
    tau = L_char**2 / alpha
    t_end = min(t_multiplier * tau, max_t)
    # Mesh spacing
    h = min(Lx / nx, Ly / ny)
    # Safe dt: CFL-ish for parabolic PDE with Backward Euler (unconditionally
    # stable, but use h²/(2α) as a guide for accuracy)
    dt_accuracy = h**2 / (2 * alpha)
    # Use ~50 time steps but cap at accuracy limit × 10 (BE allows large dt)
    dt = min(t_end / 50, dt_accuracy * 10)
    dt = max(dt, 1e-4)
    return round(t_end, 3), round(dt, 6)


def build_plan(bc_subset=None, material_subset=None, geo_subset=None):
    plans = []
    bc_types = bc_subset or [
        "dirichlet_only",
        "one_sided_flux",
        "neumann_flux_out",
        "dirichlet_robin",
        "full_robin",
        "heat_source_cool",
        "heat_source_robin",
    ]
    mats = material_subset or list(MATERIALS.keys())
    geos = [g for g in GEOMETRIES if geo_subset is None or g[0] in geo_subset]

    for bc_type in bc_types:
        for mat_name in mats:
            props = MATERIALS[mat_name]
            for geo_label, Lx, Ly, nx, ny in geos:
                t_end, dt = stable_time_params(
                    props["k"], props["rho"], props["cp"], Lx, Ly, nx, ny
                )
                source = BC_SOURCES.get(bc_type, 0.0)
                u_init = BC_UINIT.get(bc_type, 300.0)

                plans.append({
                    "run_id":  f"bcgeo_{bc_type}_{mat_name}_{geo_label}",
                    "dim": 2,
                    "nx": nx, "ny": ny,
                    "Lx": Lx, "Ly": Ly,
                    "t_end": t_end, "dt": dt,
                    "source": source,
                    "theta": 1.0,
                    "u_init": u_init,
                    "bcs": make_bcs(bc_type),
                    "bc_type": bc_type,
                    "material": mat_name,
                    "geo": geo_label,
                    **props,
                })
    return plans


def run_exists(run_id: str) -> bool:
    try:
        from database.operations import get_run
        r = get_run(run_id)
        return r is not None
    except Exception:
        return False


def run_and_store(plan: dict, kg, dry_run: bool = False) -> bool:
    run_id = plan["run_id"]

    if not dry_run and run_exists(run_id):
        print(f"  SKIP  {run_id} (already in DB)")
        return True

    print(f"  {'[DRY] ' if dry_run else ''}{run_id} "
          f"[{plan['bc_type']} | {plan['material']} | "
          f"{plan['geo']} {plan['Lx']*100:.0f}cm×{plan['Ly']*100:.0f}cm] ...",
          flush=True)

    if dry_run:
        alpha = plan["k"] / (plan["rho"] * plan["cp"])
        tau = min(plan["Lx"], plan["Ly"])**2 / alpha
        print(f"    α={alpha:.2e} m²/s  τ={tau:.1f}s  "
              f"t_end={plan['t_end']}s  dt={plan['dt']}s")
        return True

    payload = {
        "run_id":     run_id,
        "dim":        plan["dim"],
        "nx":         plan["nx"],
        "ny":         plan["ny"],
        "Lx":         plan["Lx"],
        "Ly":         plan["Ly"],
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

    try:
        create_run(run_id, plan)
    except Exception:
        pass

    t0 = time.time()
    try:
        resp = requests.post(FENICS_URL, json=payload, timeout=600)
        result = resp.json()
    except Exception as e:
        print(f"    ERROR: {e}")
        try:
            mark_run_finished(run_id, {}, RunStatus.FAILED)
        except Exception:
            pass
        return False

    elapsed = time.time() - t0

    if result.get("status") != "success":
        err = result.get("error", "unknown")[:120]
        print(f"    FAILED ({elapsed:.1f}s): {err}")
        try:
            mark_run_finished(run_id, result, RunStatus.FAILED)
        except Exception:
            pass
        return False

    try:
        mark_run_finished(run_id, result, RunStatus.SUCCESS)
    except Exception:
        pass

    warnings = []
    try:
        warnings = check_config_warnings(plan)
    except Exception:
        pass

    try:
        kg.add_run(run_id, plan, result, warnings)
        kg_ok = "✓ KG"
    except Exception as e:
        kg_ok = f"✗ KG({str(e)[:40]})"

    t_max = result.get("t_max") or result.get("max_temperature", "?")
    t_min = result.get("t_min") or result.get("min_temperature", "?")
    try:
        t_str = f"T=[{float(t_min):.0f}..{float(t_max):.0f}]K"
    except Exception:
        t_str = "T=?"
    w_str = f"  warn={[w['code'] for w in warnings]}" if warnings else ""
    print(f"    ✓ {elapsed:.1f}s  {t_str}  {kg_ok}{w_str}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="BC × Material × Geometry knowledge graph study"
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--bc",       nargs="+", help="Filter BC types")
    parser.add_argument("--material", nargs="+", help="Filter material names")
    parser.add_argument("--geo",      nargs="+",
                        help="Filter geometry labels (tiny small medium large)")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Stop after N simulations (0 = all)")
    args = parser.parse_args()

    plans = build_plan(args.bc, args.material, args.geo)
    if args.limit:
        plans = plans[:args.limit]

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}BC × Material × Geometry Study")
    print(f"{'=' * 70}")
    print(f"  Simulations planned : {len(plans)}")
    print(f"  BC types            : {sorted({p['bc_type'] for p in plans})}")
    print(f"  Materials           : {sorted({p['material'] for p in plans})}")
    print(f"  Geometries          : {sorted({p['geo'] for p in plans})}")
    print()

    # Print what the graph will look like when done
    if args.dry_run:
        print("  Sample runs:")
        for p in plans[:6]:
            run_and_store(p, None, dry_run=True)
        if len(plans) > 6:
            print(f"  ... and {len(plans)-6} more")
        print(f"\n  Total: {len(plans)} simulations")
        print(f"  Expected KG: {len({p['bc_type'] for p in plans})} BC types × "
              f"{len({p['material'] for p in plans})} materials × "
              f"{len({p['geo'] for p in plans})} geometries")
        return

    kg = get_kg()
    ok = fail = skip = 0
    for i, plan in enumerate(plans, 1):
        print(f"[{i}/{len(plans)}]", end=" ")
        success = run_and_store(plan, kg)
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n{'=' * 70}")
    print(f"  Done: {ok} succeeded, {fail} failed, {skip} skipped")
    stats = kg.stats()
    print(f"  KG stats: {stats}")


if __name__ == "__main__":
    main()
