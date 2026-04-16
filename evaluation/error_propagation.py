#!/usr/bin/env python3
"""
Error propagation analysis: quantify how wrong material properties
affect simulation outputs.

For each novel-material task, runs TWO simulations:
  1. Reference: correct properties (from KG ground truth)
  2. Wrong:     fabricated properties (from KG Off ablation)

Then computes |Delta T_max|, |Delta T_min|, |Delta T_mean| and
relative errors.  Also computes per-property sensitivity coefficients
via finite differences for the sensitivity-weighted MPF.

Run inside fenics-runner:
    docker compose exec fenics-runner python \
        /workspace/evaluation/error_propagation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulations.solvers.heat_equation import HeatConfig, HeatEquationSolver

RESULTS_DIR = Path("/workspace/evaluation/results")
OUTPUT_DIR = RESULTS_DIR / "error_propagation"

GROUND_TRUTH = {
    "Novidium":  {"k": 73.0,  "rho": 5420.0, "cp": 612.0},
    "Cryonite":  {"k": 0.42,  "rho": 1180.0, "cp": 1940.0},
    "Pyrathane": {"k": 312.0, "rho": 3850.0, "cp": 278.0},
}

TASK_MATERIAL = {
    "G1": "Novidium", "G2": "Novidium", "G3": "Novidium",
    "C1": "Cryonite", "C2": "Cryonite",
    "P1": "Pyrathane", "P2": "Pyrathane",
}

# Standardised BCs per task (same geometry/mesh/BCs, only properties differ)
TASK_CONFIGS = {
    "G1": {
        "dim": 2, "nx": 48, "ny": 48,
        "t_end": 1e-6, "dt": 1e-9, "theta": 1.0, "u_init": 350.0, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 400, "location": "left"},
            {"type": "dirichlet", "value": 300, "location": "right"},
            {"type": "neumann",   "value": 0,   "location": "top"},
            {"type": "neumann",   "value": 0,   "location": "bottom"},
        ],
    },
    "G2": {
        "dim": 2, "nx": 32, "ny": 32,
        "t_end": 200, "dt": 1.0, "theta": 1.0, "u_init": 600, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 293, "location": "left"},
            {"type": "dirichlet", "value": 293, "location": "right"},
            {"type": "dirichlet", "value": 293, "location": "top"},
            {"type": "dirichlet", "value": 293, "location": "bottom"},
        ],
    },
    "G3": {
        "dim": 2, "nx": 64, "ny": 64,
        "t_end": 150, "dt": 0.1, "theta": 1.0, "u_init": 700, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 700,          "location": "left"},
            {"type": "robin",     "alpha": 50, "u_inf": 293, "location": "right"},
            {"type": "neumann",   "value": 0,            "location": "top"},
            {"type": "neumann",   "value": 0,            "location": "bottom"},
        ],
    },
    "C1": {
        "dim": 2, "nx": 48, "ny": 48,
        "t_end": 1e-6, "dt": 1e-8, "theta": 1.0, "u_init": 315, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 350, "location": "left"},
            {"type": "dirichlet", "value": 280, "location": "right"},
            {"type": "neumann",   "value": 0,   "location": "top"},
            {"type": "neumann",   "value": 0,   "location": "bottom"},
        ],
    },
    "C2": {
        "dim": 2, "nx": 32, "ny": 32,
        "t_end": 500, "dt": 0.1, "theta": 1.0, "u_init": 325, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 400,           "location": "left"},
            {"type": "robin",     "alpha": 15, "u_inf": 250, "location": "right"},
            {"type": "neumann",   "value": 0,             "location": "top"},
            {"type": "neumann",   "value": 0,             "location": "bottom"},
        ],
    },
    "P1": {
        "dim": 2, "nx": 48, "ny": 48,
        "t_end": 1e-3, "dt": 1e-6, "theta": 1.0, "u_init": 950, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 1500, "location": "left"},
            {"type": "dirichlet", "value": 400,  "location": "right"},
            {"type": "neumann",   "value": 0,    "location": "top"},
            {"type": "neumann",   "value": 0,    "location": "bottom"},
        ],
    },
    "P2": {
        "dim": 2, "nx": 32, "ny": 32,
        "t_end": 15, "dt": 0.1, "theta": 1.0, "u_init": 2000, "source": 0,
        "bcs": [
            {"type": "dirichlet", "value": 400, "location": "left"},
            {"type": "dirichlet", "value": 400, "location": "right"},
            {"type": "dirichlet", "value": 400, "location": "top"},
            {"type": "dirichlet", "value": 400, "location": "bottom"},
        ],
    },
}

# Fabricated properties from KG Off (old model: qwen2.5-coder:32b)
KG_OFF_PROPERTIES = {
    "G1": {"k": 10,   "rho": 3000, "cp": 500},
    "G2": {"k": 45,   "rho": 8960, "cp": 137},
    "G3": {"k": 10,   "rho": 3000, "cp": 500},
    "C1": {"k": 0.15, "rho": 960,  "cp": 1370},
    "C2": {"k": 1.0,  "rho": 8960, "cp": 450},
    "P1": {"k": 0.15, "rho": 1600, "cp": 900},
    "P2": {"k": 0.15, "rho": 1600, "cp": 800},
}


def run_sim(task_id: str, props: dict, label: str) -> dict:
    """Run a simulation for a task with given material properties."""
    base = dict(TASK_CONFIGS[task_id])
    base.update(props)
    base["run_id"] = f"errprop_{task_id}_{label}"
    base["output_dir"] = str(OUTPUT_DIR / base["run_id"])
    os.makedirs(base["output_dir"], exist_ok=True)

    cfg = HeatConfig.from_dict(base)
    solver = HeatEquationSolver(cfg)
    t0 = time.perf_counter()
    result = solver.solve()
    wall = time.perf_counter() - t0

    return {
        "run_id": result.run_id,
        "T_max": float(result.max_temperature),
        "T_min": float(result.min_temperature),
        "T_mean": float(result.mean_temperature),
        "wall_time": wall,
        "status": result.status,
        "k": props["k"], "rho": props["rho"], "cp": props["cp"],
    }


def compute_sensitivity(task_id: str, ref_props: dict, ref_result: dict) -> dict:
    """Compute dT/dp for each property via central finite differences."""
    sensitivities = {}
    for prop in ("k", "rho", "cp"):
        val = ref_props[prop]
        delta = val * 0.05  # 5% perturbation
        if delta == 0:
            sensitivities[prop] = {"dTmax_dp": 0, "dTmin_dp": 0, "dTmean_dp": 0}
            continue

        props_plus = dict(ref_props)
        props_plus[prop] = val + delta
        props_minus = dict(ref_props)
        props_minus[prop] = val - delta

        r_plus = run_sim(task_id, props_plus, f"sens_{prop}_plus")
        r_minus = run_sim(task_id, props_minus, f"sens_{prop}_minus")

        sensitivities[prop] = {
            "dTmax_dp":  (r_plus["T_max"]  - r_minus["T_max"])  / (2 * delta),
            "dTmin_dp":  (r_plus["T_min"]  - r_minus["T_min"])  / (2 * delta),
            "dTmean_dp": (r_plus["T_mean"] - r_minus["T_mean"]) / (2 * delta),
        }
    return sensitivities


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    print("=" * 70)
    print("  ERROR PROPAGATION ANALYSIS")
    print("=" * 70)

    for task_id in TASK_CONFIGS:
        material = TASK_MATERIAL[task_id]
        ref_props = GROUND_TRUTH[material]
        wrong_props = KG_OFF_PROPERTIES[task_id]

        print(f"\n--- {task_id} ({material}) ---")
        print(f"  Reference:  k={ref_props['k']}, rho={ref_props['rho']}, cp={ref_props['cp']}")
        print(f"  Fabricated: k={wrong_props['k']}, rho={wrong_props['rho']}, cp={wrong_props['cp']}")

        ref_result = run_sim(task_id, ref_props, "reference")
        print(f"  Ref  -> T_max={ref_result['T_max']:.2f}, T_min={ref_result['T_min']:.2f}, "
              f"T_mean={ref_result['T_mean']:.2f} ({ref_result['wall_time']:.1f}s)")

        wrong_result = run_sim(task_id, wrong_props, "wrong")
        print(f"  Wrong-> T_max={wrong_result['T_max']:.2f}, T_min={wrong_result['T_min']:.2f}, "
              f"T_mean={wrong_result['T_mean']:.2f} ({wrong_result['wall_time']:.1f}s)")

        abs_err = {
            "dT_max":  abs(wrong_result["T_max"]  - ref_result["T_max"]),
            "dT_min":  abs(wrong_result["T_min"]  - ref_result["T_min"]),
            "dT_mean": abs(wrong_result["T_mean"] - ref_result["T_mean"]),
        }
        rel_err = {}
        for key in ("T_max", "T_min", "T_mean"):
            ref_val = ref_result[key]
            if abs(ref_val) > 1e-6:
                rel_err[f"rel_{key}"] = abs(wrong_result[key] - ref_val) / abs(ref_val)
            else:
                rel_err[f"rel_{key}"] = 0.0

        prop_errors = {
            "k_rel_err":   abs(wrong_props["k"]   - ref_props["k"])   / ref_props["k"],
            "rho_rel_err": abs(wrong_props["rho"] - ref_props["rho"]) / ref_props["rho"],
            "cp_rel_err":  abs(wrong_props["cp"]  - ref_props["cp"])  / ref_props["cp"],
        }

        print(f"  |dT_max|={abs_err['dT_max']:.2f}K  |dT_min|={abs_err['dT_min']:.2f}K  "
              f"|dT_mean|={abs_err['dT_mean']:.2f}K")
        print(f"  Rel errors: T_max={rel_err['rel_T_max']:.1%}  "
              f"T_min={rel_err['rel_T_min']:.1%}  T_mean={rel_err['rel_T_mean']:.1%}")

        # Sensitivity analysis
        print(f"  Computing sensitivities (6 extra runs)...")
        sensitivities = compute_sensitivity(task_id, ref_props, ref_result)
        for prop, s in sensitivities.items():
            print(f"    d/d{prop}: dTmax={s['dTmax_dp']:.4f}, dTmean={s['dTmean_dp']:.4f}")

        results[task_id] = {
            "material": material,
            "reference": ref_result,
            "wrong": wrong_result,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "prop_errors": prop_errors,
            "sensitivities": sensitivities,
        }

    # Save results
    out_path = RESULTS_DIR / "error_propagation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Task':<6} {'Material':<10} {'|dT_max|':>9} {'|dT_min|':>9} "
          f"{'|dT_mean|':>10} {'k_err%':>8} {'rho_err%':>9} {'cp_err%':>8}")
    print("-" * 70)
    for tid, r in results.items():
        ae = r["abs_error"]
        pe = r["prop_errors"]
        print(f"{tid:<6} {r['material']:<10} {ae['dT_max']:>9.1f} {ae['dT_min']:>9.1f} "
              f"{ae['dT_mean']:>10.1f} {pe['k_rel_err']:>7.0%} {pe['rho_rel_err']:>8.0%} "
              f"{pe['cp_rel_err']:>7.0%}")


if __name__ == "__main__":
    main()
