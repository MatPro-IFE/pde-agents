#!/usr/bin/env python3
"""
Verification & Validation runner for the heat equation solver.

Runs inside the FEniCSx container (docker compose exec fenics-runner python ...).

For each benchmark case:
  1. Runs the solver at mesh resolutions N = 8, 16, 32, 64, 128
  2. Computes the L2 error norm against the analytical solution
  3. Fits the convergence rate (expected: O(h^p) where p = 2 for P1 elements)
  4. Saves structured JSON results for paper figures/tables

Usage:
    docker compose exec fenics-runner python /workspace/evaluation/benchmarks/vv_runner.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ─── DOLFINx imports (only available inside the fenics container) ─────────────
try:
    import dolfinx
    from dolfinx.fem import Function, functionspace, form, assemble_scalar
    import ufl
    from mpi4py import MPI
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False

sys.path.insert(0, "/workspace")
from simulations.solvers.heat_equation import HeatConfig, HeatEquationSolver


RESOLUTIONS = [8, 16, 32, 64, 128]
OUTPUT_DIR = Path("/workspace/evaluation/results")

# ─── UFL-based L2 error computation ──────────────────────────────────────────

def _ufl_exact(case_name: str, mesh, t: float | None = None):
    """Return a UFL expression for the exact solution of each benchmark case.

    Using SpatialCoordinate ensures the quadrature integrates the true
    continuous error, not just the nodal error.
    """
    import ufl
    x = ufl.SpatialCoordinate(mesh)
    pi = float(np.pi)

    if case_name == "steady_linear_2d":
        # T(x,y) = x
        return x[0]

    elif case_name == "steady_sinusoidal_2d":
        # T(x,y) = sin(πx) sinh(πy) / sinh(π)
        return ufl.sin(pi * x[0]) * ufl.sinh(pi * x[1]) / float(np.sinh(pi))

    elif case_name == "transient_fourier_2d":
        # T(x,y,t) = sin(πx) sin(πy) exp(-2π² α t)
        assert t is not None
        alpha = 1.0  # k/(rho*cp)
        decay = float(np.exp(-alpha * 2.0 * pi**2 * t))
        return ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * decay

    elif case_name == "steady_source_2d":
        # T(x) = (f/2k) x(1-x) = 0.5 x(1-x)
        return 0.5 * x[0] * (1.0 - x[0])

    elif case_name == "transient_step_2d":
        # Fourier series for T(x,t) with T_L=0, T_R=1:
        # T = x + 2/π Σ_{n=1}^{N} (-1)^{n+1}/n sin(nπx) exp(-α (nπ)² t)
        assert t is not None
        alpha = 1.0
        N = 20  # enough terms for t ≥ 0.01
        u = x[0]  # steady-state part
        for n in range(1, N + 1):
            coeff = float(2.0 / (pi * n) * ((-1) ** (n + 1)))
            decay_n = float(np.exp(-alpha * (n * pi) ** 2 * t))
            u = u + coeff * ufl.sin(n * pi * x[0]) * decay_n
        return u

    else:
        raise ValueError(f"No UFL expression defined for case '{case_name}'")


@dataclass
class ConvergenceResult:
    case_name: str
    case_description: str
    resolutions: list[int]
    n_dofs: list[int]
    h_values: list[float]
    l2_errors: list[float]
    linf_errors: list[float]
    convergence_rate_l2: float
    convergence_rate_linf: float
    expected_rate: float
    wall_times: list[float]
    passed: bool
    element_degree: int


def compute_l2_error(solver: HeatEquationSolver, case_name: str,
                     t: float | None = None) -> tuple[float, float]:
    """Compute proper L2 and L∞ error norms using FEniCSx integration.

    The exact solution is defined via UFL SpatialCoordinate expressions so
    that the quadrature captures the within-element (continuous) error, not
    just the nodal error.  This is essential for cases where the FEM solution
    is nodally exact (e.g. linear or 1D-Poisson-like problems).

    Returns (l2_error, linf_error).
    """
    import ufl
    V = solver.V
    u_h = solver.u_h
    comm = solver.comm
    mesh = solver.msh

    u_exact_expr = _ufl_exact(case_name, mesh, t=t)
    error_expr = u_h - u_exact_expr

    # L2: ∫ (u_h - u_exact)² dx using exact quadrature (degree auto-selected)
    error_form = form(ufl.inner(error_expr, error_expr) * ufl.dx(
        metadata={"quadrature_degree": 8}
    ))
    l2_local = assemble_scalar(error_form)
    l2_global = comm.allreduce(l2_local, op=MPI.SUM)
    l2_error = float(np.sqrt(max(l2_global, 0.0)))

    # L∞: evaluate exact at DOF coordinates, diff with FEM values
    u_exact_nodal = Function(V)
    if t is not None:
        from evaluation.benchmarks.analytical_solutions import BENCHMARK_CASES
        fn = BENCHMARK_CASES[case_name]["analytical_fn"]
        kw = BENCHMARK_CASES[case_name]["analytical_kwargs"]
        u_exact_nodal.interpolate(lambda x: fn(x, t, **kw))
    else:
        from evaluation.benchmarks.analytical_solutions import BENCHMARK_CASES
        fn = BENCHMARK_CASES[case_name]["analytical_fn"]
        kw = BENCHMARK_CASES[case_name]["analytical_kwargs"]
        u_exact_nodal.interpolate(lambda x: fn(x, **kw))

    linf_error = float(np.max(np.abs(u_h.x.array - u_exact_nodal.x.array)))
    linf_global = comm.allreduce(linf_error, op=MPI.MAX)

    return l2_error, linf_global


def fit_convergence_rate(h_values: list[float], errors: list[float]) -> float:
    """Fit log(error) = p * log(h) + C via least squares. Returns p."""
    h_arr = np.array(h_values)
    e_arr = np.array(errors)
    mask = e_arr > 0
    if mask.sum() < 2:
        return 0.0
    log_h = np.log(h_arr[mask])
    log_e = np.log(e_arr[mask])
    p, _ = np.polyfit(log_h, log_e, 1)
    return float(p)


def run_case_steady(case_name: str, case_def: dict) -> ConvergenceResult:
    """Run a steady-state benchmark at multiple resolutions."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {case_def['name']}")
    print(f"  {case_def['description']}")
    print(f"{'='*70}")

    cfg_base = case_def["config"]
    expected_order = case_def.get("expected_order", 2)

    h_values, l2_errors, linf_errors = [], [], []
    dofs_list, times_list = [], []

    scale_dt = case_def.get("scale_dt_with_h2", False)

    for nx in RESOLUTIONS:
        cfg_dict = dict(cfg_base)
        cfg_dict["nx"] = nx
        cfg_dict["ny"] = nx
        h = cfg_dict.get("Lx", 1.0) / nx
        if scale_dt:
            # Keep temporal error ~ O(dt) << spatial error ~ O(h²)
            cfg_dict["dt"] = max(h * h / 4.0, 1e-5)
        cfg_dict["run_id"] = f"vv_{case_name}_nx{nx}"
        cfg_dict["output_dir"] = str(OUTPUT_DIR / "vv_runs")
        cfg_dict["save_every"] = 9999  # minimal I/O

        cfg = HeatConfig.from_dict(cfg_dict)
        solver = HeatEquationSolver(cfg)
        t0 = time.perf_counter()
        result = solver.solve()
        wall = time.perf_counter() - t0

        is_transient = "transient" in case_name or "step" in case_name
        if is_transient:
            l2, linf = compute_l2_error(solver, case_name, t=cfg.t_end)
        else:
            l2, linf = compute_l2_error(solver, case_name)

        h = cfg.Lx / nx
        h_values.append(h)
        l2_errors.append(l2)
        linf_errors.append(linf)
        dofs_list.append(result.n_dofs)
        times_list.append(wall)

        print(f"  nx={nx:4d}  DOFs={result.n_dofs:7,d}  h={h:.4f}  "
              f"L2={l2:.2e}  L∞={linf:.2e}  time={wall:.2f}s")

    rate_l2 = fit_convergence_rate(h_values, l2_errors)
    rate_linf = fit_convergence_rate(h_values, linf_errors)

    # For cases where P1 elements are algebraically exact (e.g. linear polynomial
    # solution), errors will be at floating-point noise level (~1e-10) with no
    # systematic h-dependence.  In that case, verify against an absolute tolerance
    # instead of a convergence rate.
    max_l2 = max(l2_errors) if l2_errors else 1.0
    exact_solution = max_l2 < 1e-6  # errors are essentially machine precision
    if exact_solution:
        passed = True
        print(f"\n  Solution is algebraically exact for P1 elements (max L2 = {max_l2:.2e})")
        print(f"  Skipping convergence rate check; verifying |e|_L2 < 1e-6  → PASS ✓")
    else:
        passed = rate_l2 >= expected_order * 0.85  # 15% tolerance
        print(f"\n  Convergence rate (L2):  {rate_l2:.2f}  (expected ≥ {expected_order})")
        print(f"  Convergence rate (L∞): {rate_linf:.2f}")
        print(f"  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return ConvergenceResult(
        case_name=case_name,
        case_description=case_def["name"],
        resolutions=RESOLUTIONS,
        n_dofs=dofs_list,
        h_values=h_values,
        l2_errors=l2_errors,
        linf_errors=linf_errors,
        convergence_rate_l2=rate_l2,
        convergence_rate_linf=rate_linf,
        expected_rate=expected_order,
        wall_times=times_list,
        passed=passed,
        element_degree=1,
    )


def run_transient_fourier_case(case_name: str, case_def: dict) -> ConvergenceResult:
    """Run the Fourier-mode transient case with custom IC."""
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {case_def['name']}")
    print(f"  {case_def['description']}")
    print(f"{'='*70}")

    cfg_base = case_def["config"]
    expected_order = case_def.get("expected_order", 2)
    ic_fn = case_def.get("ic_fn")

    h_values, l2_errors, linf_errors = [], [], []
    dofs_list, times_list = [], []

    for nx in RESOLUTIONS:
        cfg_dict = dict(cfg_base)
        cfg_dict["nx"] = nx
        cfg_dict["ny"] = nx
        # Scale dt with h² to keep temporal error subdominant
        h = cfg_dict["Lx"] / nx
        cfg_dict["dt"] = min(cfg_dict["dt"], 0.5 * h * h)
        cfg_dict["run_id"] = f"vv_{case_name}_nx{nx}"
        cfg_dict["output_dir"] = str(OUTPUT_DIR / "vv_runs")
        cfg_dict["save_every"] = 9999

        cfg = HeatConfig.from_dict(cfg_dict)
        solver = HeatEquationSolver(cfg)

        # Override the IC application to use our custom function
        solver._build_mesh()
        solver._build_function_spaces()

        if ic_fn is not None:
            solver.u_n.interpolate(ic_fn)
            solver.u_h.interpolate(ic_fn)
        else:
            solver._apply_initial_condition()

        solver._build_boundary_conditions()
        solver._build_variational_forms()

        # Manually run the time-stepping (bypass solver.solve() to keep custom IC)
        from dolfinx.fem.petsc import assemble_matrix, assemble_vector, \
            apply_lifting, set_bc
        from petsc4py import PETSc

        n_dofs = solver.V.dofmap.index_map.size_global * solver.V.dofmap.index_map_bs

        A = assemble_matrix(solver.a, bcs=solver.dirichlet_bcs)
        A.assemble()

        ksp = PETSc.KSP().create(solver.comm)
        ksp.setType(cfg.petsc_solver)
        ksp.getPC().setType(cfg.petsc_preconditioner)
        ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=300)
        ksp.setOperators(A)

        t0 = time.perf_counter()
        t_current = cfg.t_start
        n_steps = int((cfg.t_end - cfg.t_start) / cfg.dt)

        for step in range(n_steps):
            t_current += cfg.dt
            b = assemble_vector(solver.L)
            apply_lifting(b, [solver.a], bcs=[solver.dirichlet_bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, solver.dirichlet_bcs)
            ksp.solve(b, solver.u_h.x.petsc_vec)
            solver.u_h.x.scatter_forward()
            solver.u_n.x.array[:] = solver.u_h.x.array

        wall = time.perf_counter() - t0

        l2, linf = compute_l2_error(solver, case_name, t=t_current)

        h_values.append(h)
        l2_errors.append(l2)
        linf_errors.append(linf)
        dofs_list.append(n_dofs)
        times_list.append(wall)

        print(f"  nx={nx:4d}  DOFs={n_dofs:7,d}  h={h:.4f}  dt={cfg.dt:.1e}  "
              f"L2={l2:.2e}  L∞={linf:.2e}  time={wall:.2f}s")

    rate_l2 = fit_convergence_rate(h_values, l2_errors)
    rate_linf = fit_convergence_rate(h_values, linf_errors)

    max_l2 = max(l2_errors) if l2_errors else 1.0
    exact_solution = max_l2 < 1e-6
    if exact_solution:
        passed = True
        print(f"\n  Solution is algebraically exact for P1 elements (max L2 = {max_l2:.2e})")
        print(f"  Skipping convergence rate check; verifying |e|_L2 < 1e-6  → PASS ✓")
    else:
        passed = rate_l2 >= expected_order * 0.85
        print(f"\n  Convergence rate (L2):  {rate_l2:.2f}  (expected ≥ {expected_order})")
        print(f"  Convergence rate (L∞): {rate_linf:.2f}")
        print(f"  VERDICT: {'PASS ✓' if passed else 'FAIL ✗'}")

    return ConvergenceResult(
        case_name=case_name,
        case_description=case_def["name"],
        resolutions=RESOLUTIONS,
        n_dofs=dofs_list,
        h_values=h_values,
        l2_errors=l2_errors,
        linf_errors=linf_errors,
        convergence_rate_l2=rate_l2,
        convergence_rate_linf=rate_linf,
        expected_rate=expected_order,
        wall_times=times_list,
        passed=passed,
        element_degree=1,
    )


def main():
    if not DOLFINX_AVAILABLE:
        print("ERROR: DOLFINx not available. Run inside the fenics container:")
        print("  docker compose exec fenics-runner python /workspace/evaluation/benchmarks/vv_runner.py")
        sys.exit(1)

    from analytical_solutions import BENCHMARK_CASES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "vv_runs").mkdir(parents=True, exist_ok=True)

    results = {}
    total_start = time.perf_counter()

    for case_name, case_def in BENCHMARK_CASES.items():
        if case_name == "transient_fourier_2d":
            res = run_transient_fourier_case(case_name, case_def)
        elif case_def.get("note", "").startswith("Requires custom"):
            print(f"\n[SKIP] {case_name}: {case_def.get('note', '')}")
            continue
        else:
            res = run_case_steady(case_name, case_def)
        results[case_name] = asdict(res)

    total_time = time.perf_counter() - total_start

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  VERIFICATION & VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Case':<30s}  {'Rate (L2)':>10s}  {'Expected':>10s}  {'Status':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")

    all_passed = True
    for name, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_passed = False
        print(f"  {name:<30s}  {res['convergence_rate_l2']:>10.2f}  "
              f"{res['expected_rate']:>10.1f}  {status:>8s}")

    print(f"\n  Total wall time: {total_time:.1f}s")
    print(f"  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"{'='*70}\n")

    # Save results
    output_file = OUTPUT_DIR / "vv_results.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_wall_time_s": total_time,
        "all_passed": all_passed,
        "cases": results,
    }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
