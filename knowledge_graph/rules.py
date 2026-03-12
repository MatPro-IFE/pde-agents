"""
Rule-based warning engine for simulation configurations.

Pure Python — no database required. These rules encode known PDE failure modes
and physical consistency checks. They run instantly before every simulation.

Each rule is a dict with:
  code        : short identifier
  severity    : "high" | "medium" | "low"
  check(cfg)  : returns True when the rule is triggered (problem detected)
  message     : human-readable description of the problem
  recommendation : what to do instead
"""

from __future__ import annotations
from typing import Callable


# ─── Helper functions ─────────────────────────────────────────────────────────

def _min_dirichlet_value(cfg: dict) -> float | None:
    """Return the minimum Dirichlet BC value in a config, or None if no BCs."""
    bcs = cfg.get("bcs", [])
    dirichlet_vals = [
        float(bc["value"])
        for bc in bcs
        if isinstance(bc, dict) and bc.get("type") == "dirichlet"
        and bc.get("value") is not None
    ]
    return min(dirichlet_vals) if dirichlet_vals else None


def _thermal_diffusivity(cfg: dict) -> float | None:
    k   = cfg.get("k")
    rho = cfg.get("rho")
    cp  = cfg.get("cp")
    if k and rho and cp:
        return k / (rho * cp)
    return None


def _mesh_size(cfg: dict) -> float | None:
    """Characteristic mesh element size from nx, domain assumed unit by default."""
    nx = cfg.get("nx")
    ny = cfg.get("ny")
    if nx and ny:
        lx = cfg.get("lx", 1.0)
        ly = cfg.get("ly", 1.0)
        return max(lx / nx, ly / ny)
    return None


# ─── Rule definitions ─────────────────────────────────────────────────────────

RULES: list[dict] = [

    # ── Inconsistent initial condition ──────────────────────────────────────
    {
        "code": "INCONSISTENT_IC",
        "severity": "high",
        "check": lambda cfg: (
            _min_dirichlet_value(cfg) is not None
            and abs(cfg.get("u_init", 0.0) - _min_dirichlet_value(cfg)) > 100.0
        ),
        "message": (
            "u_init={u_init:.0f} K differs significantly from the minimum "
            "Dirichlet BC value ({min_bc:.0f} K). "
            "This causes a large initial jump that produces Gibbs-like overshoot."
        ),
        "recommendation": (
            "Set u_init close to the minimum boundary temperature. "
            "For a plate with T_left=800 K and T_right=300 K, use u_init=300 or 800."
        ),
    },

    # ── CFL-like stability for explicit theta=0 ─────────────────────────────
    {
        "code": "EXPLICIT_CFL_VIOLATION",
        "severity": "high",
        "check": lambda cfg: (
            cfg.get("theta", 1.0) < 0.5
            and _thermal_diffusivity(cfg) is not None
            and _mesh_size(cfg) is not None
            and cfg.get("dt", 1.0) > (
                _mesh_size(cfg) ** 2 / (2 * _thermal_diffusivity(cfg))
            )
        ),
        "message": (
            "dt={dt} exceeds the CFL stability limit for theta={theta}. "
            "Explicit or near-explicit schemes require dt ≤ h²/(2α)."
        ),
        "recommendation": (
            "Reduce dt or switch to the implicit scheme (theta=1.0 for backward Euler, "
            "theta=0.5 for Crank-Nicolson). Implicit schemes are unconditionally stable."
        ),
    },

    # ── Near-explicit scheme warning ────────────────────────────────────────
    {
        "code": "NEAR_EXPLICIT_SCHEME",
        "severity": "medium",
        "check": lambda cfg: 0.0 <= cfg.get("theta", 1.0) < 0.5,
        "message": "theta={theta} < 0.5 gives a conditionally stable scheme.",
        "recommendation": (
            "Use theta=0.5 (Crank-Nicolson) for second-order accuracy with "
            "unconditional stability, or theta=1.0 (backward Euler) for robustness."
        ),
    },

    # ── Very coarse mesh ────────────────────────────────────────────────────
    {
        "code": "COARSE_MESH_2D",
        "severity": "medium",
        "check": lambda cfg: (
            cfg.get("dim") == 2
            and (cfg.get("nx", 20) < 10 or cfg.get("ny", 20) < 10)
        ),
        "message": "nx={nx}, ny={ny} — mesh is very coarse for a 2D simulation.",
        "recommendation": "Use at least nx=ny=20 for meaningful 2D results.",
    },

    {
        "code": "COARSE_MESH_3D",
        "severity": "medium",
        "check": lambda cfg: (
            cfg.get("dim") == 3
            and (
                cfg.get("nx", 10) < 8
                or cfg.get("ny", 10) < 8
                or cfg.get("nz", 10) < 8
            )
        ),
        "message": "nx={nx}, ny={ny}, nz={nz} — mesh is very coarse for a 3D simulation.",
        "recommendation": (
            "Use at least 10 elements per direction for 3D. "
            "Note: 3D DOF count scales as nx·ny·nz — double each dimension = 8× cost."
        ),
    },

    # ── Very short simulation time ───────────────────────────────────────────
    {
        "code": "SHORT_SIMULATION",
        "severity": "low",
        "check": lambda cfg: (
            cfg.get("t_end", 1.0) < cfg.get("dt", 0.1) * 5
        ),
        "message": "t_end={t_end} gives fewer than 5 time steps with dt={dt}.",
        "recommendation": (
            "Increase t_end or decrease dt to allow the solution to evolve. "
            "Steady-state problems need t_end ≈ L²/α (characteristic diffusion time)."
        ),
    },

    # ── Negative or zero material properties ────────────────────────────────
    {
        "code": "INVALID_MATERIAL_PROPS",
        "severity": "high",
        "check": lambda cfg: any([
            (cfg.get("k") is not None and cfg["k"] <= 0),
            (cfg.get("rho") is not None and cfg["rho"] <= 0),
            (cfg.get("cp") is not None and cfg["cp"] <= 0),
        ]),
        "message": "Material properties contain non-positive values: k={k}, rho={rho}, cp={cp}.",
        "recommendation": "k, rho, cp must all be strictly positive physical values.",
    },

    # ── Very large time step relative to diffusion timescale ────────────────
    {
        "code": "LARGE_DT_RELATIVE_TO_DIFFUSION",
        "severity": "low",
        "check": lambda cfg: (
            _thermal_diffusivity(cfg) is not None
            and _mesh_size(cfg) is not None
            and cfg.get("dt", 0.1) > 10 * (_mesh_size(cfg) ** 2 / _thermal_diffusivity(cfg))
        ),
        "message": "dt={dt} is very large relative to the mesh diffusion timescale.",
        "recommendation": (
            "Large dt may smear transient behavior. "
            "If steady state is the goal this is acceptable; "
            "otherwise reduce dt to resolve the temperature evolution."
        ),
    },

    # ── No boundary conditions ───────────────────────────────────────────────
    {
        "code": "NO_BOUNDARY_CONDITIONS",
        "severity": "high",
        "check": lambda cfg: not cfg.get("bcs"),
        "message": "No boundary conditions specified.",
        "recommendation": (
            "Add at least one Dirichlet BC. "
            "A pure Neumann problem is ill-posed without a reference temperature."
        ),
    },
]


# ─── Public API ───────────────────────────────────────────────────────────────

def check_config(cfg: dict) -> list[dict]:
    """
    Run all rules against a simulation config.

    Returns a list of triggered warnings, each with:
      code, severity, message (with values filled in), recommendation
    """
    triggered = []
    for rule in RULES:
        try:
            if rule["check"](cfg):
                # Fill placeholders in message with actual config values
                msg = rule["message"].format(
                    u_init=cfg.get("u_init", 0),
                    min_bc=_min_dirichlet_value(cfg) or 0,
                    dt=cfg.get("dt", "?"),
                    theta=cfg.get("theta", 1.0),
                    nx=cfg.get("nx", "?"),
                    ny=cfg.get("ny", "?"),
                    nz=cfg.get("nz", "?"),
                    t_end=cfg.get("t_end", "?"),
                    k=cfg.get("k", "?"),
                    rho=cfg.get("rho", "?"),
                    cp=cfg.get("cp", "?"),
                )
                triggered.append({
                    "code":           rule["code"],
                    "severity":       rule["severity"],
                    "message":        msg,
                    "recommendation": rule["recommendation"],
                })
        except Exception:
            pass  # Never let a rule crash the agent
    return triggered
