#!/usr/bin/env python3
"""
Full Parameter Sweep — 805+ simulation runs across 8 studies
=============================================================

Studies
-------
  s1  2D rectangle, built-in mesh          7 BC × 10 materials × 4 domains     = 280
  s2  2D Gmsh complex geometries           6 shapes × 2 mesh × 4 mat × 3 BC    = 144
  s3  3D Gmsh geometries (box, cylinder)   2 shapes × 2 mesh × 4 mat × 4 BC    =  64
  s4  Mesh-refinement convergence          5 nx × 3 materials × 3 BC            =  45
  s5  Initial condition sweep              6 IC × 4 BC × 4 materials            =  96
  s6  Theta-scheme (time integration)      4 θ × 4 materials × 4 BC             =  64
  s7  Convection-coefficient parametric    6 h × 6 materials × 2 domains        =  72
  s8  Internal heat-source intensity       5 Q × 4 materials × 2 domains        =  40
                                                                         TOTAL   805

Features
--------
- Idempotent: skips run_ids already in the database.
- Physics-stable dt / t_end: scaled to material diffusivity + domain size.
- Stores every result in PostgreSQL and Neo4j knowledge graph.
- Progress reporting every 50 runs (rate, ETA).

Usage
-----
  # Dry-run (print plan, exit)
  docker exec pde-agents python3 /app/scripts/sweep_full_study.py --dry-run

  # All studies in background
  docker exec -d pde-agents bash -c \\
    "python3 /app/scripts/sweep_full_study.py > /tmp/sweep.log 2>&1"

  # Monitor
  docker exec pde-agents tail -f /tmp/sweep.log

  # Single study
  docker exec pde-agents python3 /app/scripts/sweep_full_study.py --study s2 s3

  # First N runs only (useful for quick smoke-test)
  docker exec pde-agents python3 /app/scripts/sweep_full_study.py --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta

import requests

sys.path.insert(0, "/app")

from database.models import RunStatus
from database.operations import create_run, mark_run_finished
from knowledge_graph.graph import get_kg
from knowledge_graph.rules import check_config as check_config_warnings

FENICS_URL = "http://fenics-runner:8080/run"

# ─── Material library ─────────────────────────────────────────────────────────
# k [W/(m·K)], rho [kg/m³], cp [J/(kg·K)]

MATERIALS: dict[str, dict] = {
    "copper":       {"k": 385.0, "rho": 8_960.0, "cp":  385.0},
    "aluminium":    {"k": 200.0, "rho": 2_700.0, "cp":  900.0},
    "silicon":      {"k": 150.0, "rho": 2_330.0, "cp":  700.0},
    "steel":        {"k":  50.0, "rho": 7_800.0, "cp":  500.0},
    "stainless316": {"k":  16.0, "rho": 8_000.0, "cp":  500.0},
    "titanium":     {"k":   6.7, "rho": 4_430.0, "cp":  526.0},
    "concrete":     {"k":   1.7, "rho": 2_300.0, "cp":  880.0},
    "glass":        {"k":   1.0, "rho": 2_230.0, "cp":  830.0},
    "water":        {"k":   0.6, "rho": 1_000.0, "cp": 4_182.0},
    "air":          {"k": 0.026, "rho":     1.2,  "cp": 1_005.0},
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _alpha(mat: dict) -> float:
    """Thermal diffusivity [m²/s]."""
    return mat["k"] / (mat["rho"] * mat["cp"])


def stable_time_params(
    mat: dict,
    L_char: float,
    n_steps: int = 50,
    t_mult: float = 3.0,
    max_t: float = 10_000.0,
) -> tuple[float, float]:
    """
    Return (t_end, dt) that give ~n_steps time steps and reach ~95 % of
    steady-state for material *mat* over characteristic length *L_char*.
    """
    a = _alpha(mat)
    tau   = L_char ** 2 / a
    t_end = min(t_mult * tau, max_t)
    dt    = min(t_end / n_steps, (L_char / n_steps) ** 2 / (2 * a) * 10)
    dt    = max(dt, 1e-6)
    return round(t_end, 4), round(dt, 8)


def _ins(loc: str) -> dict:
    """Insulated (zero-flux Neumann) BC on *loc*."""
    return {"type": "neumann", "value": 0.0, "location": loc}


def run_exists(run_id: str) -> bool:
    try:
        from database.operations import get_run
        return get_run(run_id) is not None
    except Exception:
        return False


# ─── Study 1 — 2D rectangle, built-in mesh ────────────────────────────────────
# 7 BC types × 10 materials × 4 domain sizes = 280 runs

_BC_NAMES_S1 = [
    "dirichlet_only",
    "one_sided_flux",
    "neumann_flux_out",
    "dirichlet_robin",
    "full_robin",
    "heat_source_cool",
    "heat_source_robin",
]

_DOMAINS_S1 = [
    # (label,  Lx,   Ly,   nx, ny)
    ("tiny",   0.01, 0.01, 16, 16),
    ("small",  0.04, 0.02, 40, 20),
    ("medium", 0.10, 0.05, 50, 25),
    ("large",  0.30, 0.10, 60, 20),
]

_S1_SOURCES = {"heat_source_cool": 10_000.0, "heat_source_robin": 20_000.0}
_S1_UINIT   = {"full_robin": 293.0, "heat_source_cool": 300.0, "heat_source_robin": 293.0}


def _make_bcs_2d_rect(
    bc_type: str,
    T_hot: float = 800.0,
    T_cold: float = 300.0,
    h: float = 25.0,
    T_amb: float = 293.0,
    q: float = 5_000.0,
) -> list:
    configs: dict[str, list] = {
        "dirichlet_only": [
            {"type": "dirichlet", "value": T_hot,  "location": "left"},
            {"type": "dirichlet", "value": T_cold, "location": "right"},
            _ins("top"), _ins("bottom"),
        ],
        "one_sided_flux": [
            {"type": "dirichlet", "value": T_hot, "location": "left"},
            _ins("right"), _ins("top"), _ins("bottom"),
        ],
        "neumann_flux_out": [
            {"type": "dirichlet", "value": T_hot, "location": "left"},
            {"type": "neumann",   "value": q,     "location": "right"},
            _ins("top"), _ins("bottom"),
        ],
        "dirichlet_robin": [
            {"type": "dirichlet", "value": T_hot,  "location": "left"},
            {"type": "dirichlet", "value": T_cold, "location": "right"},
            {"type": "robin", "alpha": h, "u_inf": T_amb, "location": "top"},
            {"type": "robin", "alpha": h, "u_inf": T_amb, "location": "bottom"},
        ],
        "full_robin": [
            {"type": "robin", "alpha": h, "u_inf": T_amb, "location": loc}
            for loc in ("left", "right", "top", "bottom")
        ],
        "heat_source_cool": [
            {"type": "dirichlet", "value": T_cold, "location": "left"},
            {"type": "dirichlet", "value": T_cold, "location": "right"},
            _ins("top"), _ins("bottom"),
        ],
        "heat_source_robin": [
            {"type": "robin", "alpha": h * 4, "u_inf": T_amb, "location": loc}
            for loc in ("left", "right", "top", "bottom")
        ],
    }
    return configs[bc_type]


def study1() -> list[dict]:
    plans: list[dict] = []
    for bc_type in _BC_NAMES_S1:
        for mat_name, mat in MATERIALS.items():
            for geo_label, Lx, Ly, nx, ny in _DOMAINS_S1:
                t_end, dt = stable_time_params(mat, min(Lx, Ly))
                plans.append({
                    "run_id":   f"s1_{bc_type}_{mat_name}_{geo_label}",
                    "dim": 2, "nx": nx, "ny": ny, "Lx": Lx, "Ly": Ly,
                    "t_end": t_end, "dt": dt, "theta": 1.0,
                    "source": _S1_SOURCES.get(bc_type, 0.0),
                    "u_init": _S1_UINIT.get(bc_type, 300.0),
                    "bcs":   _make_bcs_2d_rect(bc_type),
                    "_bc": bc_type, "_mat": mat_name, "_geo": geo_label,
                    **mat,
                })
    return plans


# ─── Study 2 — 2D Gmsh complex geometries ─────────────────────────────────────
# 6 shapes × 2 mesh sizes × 4 materials × 3 BC patterns = 144 runs

_S2_MATS = ["copper", "steel", "concrete", "air"]

_GMSH_2D_SPECS: list[dict] = [
    {
        "type": "l_shape",
        "coarse": {"Lx": 0.1, "Ly": 0.1, "cut_x": 0.04, "cut_y": 0.04, "mesh_size": 0.008},
        "fine":   {"Lx": 0.1, "Ly": 0.1, "cut_x": 0.04, "cut_y": 0.04, "mesh_size": 0.004},
        "L_char": 0.1,
        "bc_patterns": [
            ("left_hot_right_cold",
             [{"type": "dirichlet", "value": 800.0, "location": "left"},
              {"type": "dirichlet", "value": 300.0, "location": "right"},
              _ins("bottom"), _ins("inner_h"), _ins("inner_v"), _ins("top")],
             0.0, 300.0),
            ("full_robin",
             [{"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": loc}
              for loc in ("left", "bottom", "right", "inner_h", "inner_v", "top")],
             0.0, 700.0),
            ("source_robin",
             [{"type": "robin", "alpha": 100.0, "u_inf": 293.0, "location": loc}
              for loc in ("left", "bottom", "right", "inner_h", "inner_v", "top")],
             10_000.0, 293.0),
        ],
    },
    {
        "type": "annulus",
        "coarse": {"r_in": 0.02, "r_out": 0.06, "mesh_size": 0.006},
        "fine":   {"r_in": 0.02, "r_out": 0.06, "mesh_size": 0.003},
        "L_char": 0.04,
        "bc_patterns": [
            ("inner_hot_outer_cold",
             [{"type": "dirichlet", "value": 800.0, "location": "inner_wall"},
              {"type": "dirichlet", "value": 300.0, "location": "outer_wall"}],
             0.0, 300.0),
            ("inner_hot_outer_robin",
             [{"type": "dirichlet", "value": 800.0, "location": "inner_wall"},
              {"type": "robin", "alpha": 50.0, "u_inf": 300.0, "location": "outer_wall"}],
             0.0, 300.0),
            ("source_both_robin",
             [{"type": "robin", "alpha": 100.0, "u_inf": 293.0, "location": "inner_wall"},
              {"type": "robin", "alpha": 25.0,  "u_inf": 293.0, "location": "outer_wall"}],
             10_000.0, 293.0),
        ],
    },
    {
        "type": "hollow_rectangle",
        "coarse": {"Lx": 0.1, "Ly": 0.08, "hole_w": 0.04, "hole_h": 0.03, "mesh_size": 0.007},
        "fine":   {"Lx": 0.1, "Ly": 0.08, "hole_w": 0.04, "hole_h": 0.03, "mesh_size": 0.003},
        "L_char": 0.08,
        "bc_patterns": [
            ("outer_hot_hole_cold",
             [{"type": "dirichlet", "value": 800.0, "location": loc}
              for loc in ("left", "right", "top", "bottom")]
             + [{"type": "dirichlet", "value": 300.0, "location": "hole_wall"}],
             0.0, 300.0),
            ("left_hot_hole_robin",
             [{"type": "dirichlet", "value": 800.0, "location": "left"},
              {"type": "dirichlet", "value": 300.0, "location": "right"},
              _ins("top"), _ins("bottom"),
              {"type": "robin", "alpha": 100.0, "u_inf": 293.0, "location": "hole_wall"}],
             0.0, 300.0),
            ("source_all_robin",
             [{"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": loc}
              for loc in ("left", "right", "top", "bottom", "hole_wall")],
             5_000.0, 293.0),
        ],
    },
    {
        "type": "t_shape",
        "coarse": {"flange_w": 0.1, "flange_h": 0.02, "web_w": 0.04, "web_h": 0.08, "mesh_size": 0.006},
        "fine":   {"flange_w": 0.1, "flange_h": 0.02, "web_w": 0.04, "web_h": 0.08, "mesh_size": 0.003},
        "L_char": 0.08,
        "bc_patterns": [
            ("bottom_hot_top_robin",
             [{"type": "dirichlet", "value": 800.0, "location": "bottom"},
              {"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": "top"},
              _ins("left"), _ins("right"), _ins("inner_left"), _ins("inner_right")],
             0.0, 300.0),
            ("bottom_hot_top_cold_step_conv",
             [{"type": "dirichlet", "value": 800.0, "location": "bottom"},
              {"type": "dirichlet", "value": 300.0, "location": "top"},
              {"type": "robin", "alpha": 50.0, "u_inf": 293.0, "location": "inner_left"},
              {"type": "robin", "alpha": 50.0, "u_inf": 293.0, "location": "inner_right"},
              _ins("left"), _ins("right")],
             0.0, 300.0),
            ("source_full_robin",
             [{"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": loc}
              for loc in ("bottom", "top", "left", "right", "inner_left", "inner_right")],
             5_000.0, 293.0),
        ],
    },
    {
        "type": "stepped_notch",
        "coarse": {"Lx": 0.12, "Ly": 0.06, "step_x": 0.07, "step_h": 0.025, "mesh_size": 0.006},
        "fine":   {"Lx": 0.12, "Ly": 0.06, "step_x": 0.07, "step_h": 0.025, "mesh_size": 0.003},
        "L_char": 0.06,
        "bc_patterns": [
            ("left_hot_right_cold",
             [{"type": "dirichlet", "value": 800.0, "location": "left"},
              {"type": "dirichlet", "value": 300.0, "location": "right_lower"},
              {"type": "dirichlet", "value": 300.0, "location": "right_upper"},
              _ins("bottom"), _ins("top"), _ins("step_face"), _ins("step_riser")],
             0.0, 300.0),
            ("left_hot_right_robin_step_conv",
             [{"type": "dirichlet", "value": 800.0, "location": "left"},
              {"type": "robin", "alpha": 50.0, "u_inf": 300.0, "location": "right_lower"},
              {"type": "robin", "alpha": 50.0, "u_inf": 300.0, "location": "right_upper"},
              {"type": "robin", "alpha": 150.0, "u_inf": 293.0, "location": "step_face"},
              _ins("bottom"), _ins("top"), _ins("step_riser")],
             0.0, 300.0),
            ("source_full_robin",
             [{"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": loc}
              for loc in ("left", "right_lower", "right_upper", "bottom", "top",
                          "step_face", "step_riser")],
             10_000.0, 293.0),
        ],
    },
    {
        "type": "circle",
        "coarse": {"radius": 0.05, "mesh_size": 0.006},
        "fine":   {"radius": 0.05, "mesh_size": 0.003},
        "L_char": 0.05,
        "bc_patterns": [
            # Hot interior, cool via convection — study cooling curve
            ("cooling_from_init",
             [{"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": "wall"}],
             0.0, 800.0),
            # Fixed hot wall — study transient from cold interior
            ("wall_fixed_hot",
             [{"type": "dirichlet", "value": 800.0, "location": "wall"}],
             0.0, 300.0),
            # Internal source with convective wall
            ("source_wall_robin",
             [{"type": "robin", "alpha": 100.0, "u_inf": 293.0, "location": "wall"}],
             50_000.0, 293.0),
        ],
    },
]


def study2() -> list[dict]:
    plans: list[dict] = []
    for spec in _GMSH_2D_SPECS:
        geo_type = spec["type"]
        L_char   = spec["L_char"]
        for mesh_label, geo_params in (("coarse", spec["coarse"]), ("fine", spec["fine"])):
            for mat_name in _S2_MATS:
                mat = MATERIALS[mat_name]
                for bc_name, bcs, source, u_init in spec["bc_patterns"]:
                    t_end, dt = stable_time_params(mat, L_char)
                    plans.append({
                        "run_id":   f"s2_{geo_type}_{mesh_label}_{mat_name}_{bc_name}",
                        "dim": 2,
                        "geometry": {"type": geo_type, **geo_params},
                        "t_end": t_end, "dt": dt, "theta": 1.0,
                        "source": source, "u_init": u_init,
                        "bcs":   bcs,
                        "_bc": bc_name, "_mat": mat_name, "_geo": geo_type,
                        **mat,
                    })
    return plans


# ─── Study 3 — 3D Gmsh geometries ─────────────────────────────────────────────
# 2 shapes × 2 mesh sizes × 4 materials × 4 BC patterns = 64 runs

_S3_MATS = ["copper", "steel", "concrete", "air"]

_GMSH_3D_SPECS: list[dict] = [
    {
        "type": "box",
        "coarse": {"Lx": 0.1, "Ly": 0.1, "Lz": 0.05, "mesh_size": 0.015},
        "fine":   {"Lx": 0.1, "Ly": 0.1, "Lz": 0.05, "mesh_size": 0.008},
        "L_char": 0.05,
        "bc_patterns": [
            ("bottom_hot_top_cold",
             [{"type": "dirichlet", "value": 800.0, "location": "bottom"},
              {"type": "dirichlet", "value": 300.0, "location": "top"},
              _ins("left"), _ins("right"), _ins("front"), _ins("back")],
             0.0, 300.0),
            ("left_hot_right_cold",
             [{"type": "dirichlet", "value": 800.0, "location": "left"},
              {"type": "dirichlet", "value": 300.0, "location": "right"},
              _ins("front"), _ins("back"), _ins("bottom"), _ins("top")],
             0.0, 300.0),
            ("all_robin",
             [{"type": "robin", "alpha": 25.0, "u_inf": 293.0, "location": loc}
              for loc in ("left", "right", "front", "back", "bottom", "top")],
             0.0, 700.0),
            ("source_all_robin",
             [{"type": "robin", "alpha": 50.0, "u_inf": 293.0, "location": loc}
              for loc in ("left", "right", "front", "back", "bottom", "top")],
             5_000.0, 293.0),
        ],
    },
    {
        "type": "cylinder",
        "coarse": {"radius": 0.04, "height": 0.1, "mesh_size": 0.012},
        "fine":   {"radius": 0.04, "height": 0.1, "mesh_size": 0.006},
        "L_char": 0.04,
        "bc_patterns": [
            ("bottom_hot_top_cold_lat_ins",
             [{"type": "dirichlet", "value": 800.0, "location": "bottom_face"},
              {"type": "dirichlet", "value": 300.0, "location": "top_face"},
              _ins("lateral_wall")],
             0.0, 300.0),
            ("bottom_hot_top_robin_lat_robin",
             [{"type": "dirichlet", "value": 800.0, "location": "bottom_face"},
              {"type": "robin", "alpha": 50.0, "u_inf": 300.0, "location": "top_face"},
              {"type": "robin", "alpha": 25.0, "u_inf": 300.0, "location": "lateral_wall"}],
             0.0, 300.0),
            ("lateral_fixed_bot_hot",
             [{"type": "dirichlet", "value": 300.0, "location": "lateral_wall"},
              {"type": "dirichlet", "value": 800.0, "location": "bottom_face"},
              _ins("top_face")],
             0.0, 300.0),
            ("source_all_robin",
             [{"type": "robin", "alpha": 50.0, "u_inf": 293.0, "location": loc}
              for loc in ("lateral_wall", "bottom_face", "top_face")],
             5_000.0, 293.0),
        ],
    },
]


def study3() -> list[dict]:
    plans: list[dict] = []
    for spec in _GMSH_3D_SPECS:
        geo_type = spec["type"]
        L_char   = spec["L_char"]
        for mesh_label, geo_params in (("coarse", spec["coarse"]), ("fine", spec["fine"])):
            for mat_name in _S3_MATS:
                mat = MATERIALS[mat_name]
                for bc_name, bcs, source, u_init in spec["bc_patterns"]:
                    t_end, dt = stable_time_params(mat, L_char)
                    plans.append({
                        "run_id":   f"s3_{geo_type}_{mesh_label}_{mat_name}_{bc_name}",
                        "dim": 3,
                        "geometry": {"type": geo_type, **geo_params},
                        "t_end": t_end, "dt": dt, "theta": 1.0,
                        "source": source, "u_init": u_init,
                        "bcs":   bcs,
                        "_bc": bc_name, "_mat": mat_name, "_geo": geo_type,
                        **mat,
                    })
    return plans


# ─── Study 4 — Mesh-refinement convergence ────────────────────────────────────
# 5 nx values × 3 materials × 3 BC types = 45 runs

_S4_NX     = [8, 16, 32, 64, 96]
_S4_MATS   = ["copper", "steel", "air"]
_S4_BCS    = ["dirichlet_only", "dirichlet_robin", "full_robin"]
_S4_Lx, _S4_Ly = 0.10, 0.05


def study4() -> list[dict]:
    plans: list[dict] = []
    for nx in _S4_NX:
        ny = max(8, nx // 2)
        for mat_name in _S4_MATS:
            mat = MATERIALS[mat_name]
            t_end, dt = stable_time_params(mat, min(_S4_Lx, _S4_Ly))
            for bc_type in _S4_BCS:
                plans.append({
                    "run_id":  f"s4_{mat_name}_{bc_type}_nx{nx}",
                    "dim": 2, "nx": nx, "ny": ny, "Lx": _S4_Lx, "Ly": _S4_Ly,
                    "t_end": t_end, "dt": dt, "theta": 1.0,
                    "source": _S1_SOURCES.get(bc_type, 0.0),
                    "u_init": _S1_UINIT.get(bc_type, 300.0),
                    "bcs":   _make_bcs_2d_rect(bc_type),
                    "_bc": bc_type, "_mat": mat_name, "_geo": f"medium_nx{nx}",
                    **mat,
                })
    return plans


# ─── Study 5 — Initial condition sweep ────────────────────────────────────────
# 6 IC values × 4 BC types × 4 materials = 96 runs

_S5_IC     = [0.0, 200.0, 300.0, 450.0, 600.0, 800.0]
_S5_BCS    = ["dirichlet_only", "dirichlet_robin", "full_robin", "heat_source_robin"]
_S5_MATS   = ["copper", "steel", "concrete", "air"]
_S5_Lx, _S5_Ly, _S5_nx, _S5_ny = 0.10, 0.05, 50, 25


def study5() -> list[dict]:
    plans: list[dict] = []
    for u_init in _S5_IC:
        ic_tag = f"ic{int(u_init)}"
        for mat_name in _S5_MATS:
            mat = MATERIALS[mat_name]
            t_end, dt = stable_time_params(mat, min(_S5_Lx, _S5_Ly))
            for bc_type in _S5_BCS:
                plans.append({
                    "run_id":  f"s5_{ic_tag}_{mat_name}_{bc_type}",
                    "dim": 2, "nx": _S5_nx, "ny": _S5_ny,
                    "Lx": _S5_Lx, "Ly": _S5_Ly,
                    "t_end": t_end, "dt": dt, "theta": 1.0,
                    "source": _S1_SOURCES.get(bc_type, 0.0),
                    "u_init": u_init,
                    "bcs":   _make_bcs_2d_rect(bc_type),
                    "_bc": bc_type, "_mat": mat_name, "_geo": "medium",
                    **mat,
                })
    return plans


# ─── Study 6 — Theta-scheme (time integration accuracy) ───────────────────────
# 4 θ values × 4 materials × 4 BC types = 64 runs

_S6_THETA  = [0.5, 0.67, 0.75, 1.0]
_S6_MATS   = ["copper", "steel", "concrete", "air"]
_S6_BCS    = ["dirichlet_only", "dirichlet_robin", "full_robin", "heat_source_cool"]
_S6_Lx, _S6_Ly, _S6_nx, _S6_ny = 0.10, 0.05, 50, 25


def study6() -> list[dict]:
    plans: list[dict] = []
    for theta in _S6_THETA:
        th_tag = f"th{int(theta * 100)}"
        for mat_name in _S6_MATS:
            mat = MATERIALS[mat_name]
            t_end, dt = stable_time_params(mat, min(_S6_Lx, _S6_Ly))
            # Crank-Nicolson benefits from a finer dt
            dt_use = round(dt / 2, 8) if theta <= 0.5 else dt
            for bc_type in _S6_BCS:
                plans.append({
                    "run_id":  f"s6_{th_tag}_{mat_name}_{bc_type}",
                    "dim": 2, "nx": _S6_nx, "ny": _S6_ny,
                    "Lx": _S6_Lx, "Ly": _S6_Ly,
                    "t_end": t_end, "dt": dt_use, "theta": theta,
                    "source": _S1_SOURCES.get(bc_type, 0.0),
                    "u_init": _S1_UINIT.get(bc_type, 300.0),
                    "bcs":   _make_bcs_2d_rect(bc_type),
                    "_bc": bc_type, "_mat": mat_name, "_geo": "medium",
                    **mat,
                })
    return plans


# ─── Study 7 — Convection-coefficient parametric ──────────────────────────────
# 6 h values × 6 materials × 2 domain sizes = 72 runs
# All-Robin BCs.  Initial condition = 700 K (cooling study).

_S7_H      = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0]
_S7_MATS   = ["copper", "aluminium", "steel", "titanium", "concrete", "air"]
_S7_DOMAINS = [
    ("medium", 0.10, 0.05, 50, 25),
    ("large",  0.30, 0.10, 60, 20),
]


def study7() -> list[dict]:
    plans: list[dict] = []
    for h_val in _S7_H:
        h_tag = f"h{int(h_val)}"
        for mat_name in _S7_MATS:
            mat = MATERIALS[mat_name]
            for geo_label, Lx, Ly, nx, ny in _S7_DOMAINS:
                t_end, dt = stable_time_params(mat, min(Lx, Ly))
                bcs = [
                    {"type": "robin", "alpha": h_val, "u_inf": 293.0, "location": loc}
                    for loc in ("left", "right", "top", "bottom")
                ]
                plans.append({
                    "run_id":  f"s7_{h_tag}_{mat_name}_{geo_label}",
                    "dim": 2, "nx": nx, "ny": ny, "Lx": Lx, "Ly": Ly,
                    "t_end": t_end, "dt": dt, "theta": 1.0,
                    "source": 0.0, "u_init": 700.0,
                    "bcs":   bcs,
                    "_bc": f"full_robin_{h_tag}", "_mat": mat_name, "_geo": geo_label,
                    **mat,
                })
    return plans


# ─── Study 8 — Internal heat-source intensity ─────────────────────────────────
# 5 source values × 4 materials × 2 domain sizes = 40 runs
# Robin cooling on all sides.

_S8_SOURCES = [100.0, 1_000.0, 5_000.0, 10_000.0, 50_000.0]
_S8_MATS    = ["copper", "steel", "concrete", "air"]
_S8_DOMAINS = [
    ("small",  0.04, 0.02, 40, 20),
    ("medium", 0.10, 0.05, 50, 25),
]


def study8() -> list[dict]:
    plans: list[dict] = []
    for src in _S8_SOURCES:
        src_tag = f"q{int(src)}"
        for mat_name in _S8_MATS:
            mat = MATERIALS[mat_name]
            for geo_label, Lx, Ly, nx, ny in _S8_DOMAINS:
                t_end, dt = stable_time_params(mat, min(Lx, Ly))
                bcs = [
                    {"type": "robin", "alpha": 100.0, "u_inf": 293.0, "location": loc}
                    for loc in ("left", "right", "top", "bottom")
                ]
                plans.append({
                    "run_id":  f"s8_{src_tag}_{mat_name}_{geo_label}",
                    "dim": 2, "nx": nx, "ny": ny, "Lx": Lx, "Ly": Ly,
                    "t_end": t_end, "dt": dt, "theta": 1.0,
                    "source": src, "u_init": 293.0,
                    "bcs":   bcs,
                    "_bc": "source_robin", "_mat": mat_name, "_geo": geo_label,
                    **mat,
                })
    return plans


# ─── Study registry ───────────────────────────────────────────────────────────

STUDY_FUNCS: dict[str, callable] = {
    "s1": study1,
    "s2": study2,
    "s3": study3,
    "s4": study4,
    "s5": study5,
    "s6": study6,
    "s7": study7,
    "s8": study8,
}

STUDY_DESCRIPTIONS: dict[str, str] = {
    "s1": "2D rectangle built-in mesh  (7 BC × 10 mat × 4 domain)",
    "s2": "2D Gmsh complex geometries  (6 shapes × 2 mesh × 4 mat × 3 BC)",
    "s3": "3D Gmsh geometries          (2 shapes × 2 mesh × 4 mat × 4 BC)",
    "s4": "Mesh-refinement convergence (5 nx × 3 mat × 3 BC)",
    "s5": "Initial condition sweep     (6 IC × 4 BC × 4 mat)",
    "s6": "Theta-scheme                (4 θ × 4 mat × 4 BC)",
    "s7": "Convection-h parametric     (6 h × 6 mat × 2 domain)",
    "s8": "Heat-source intensity       (5 Q × 4 mat × 2 domain)",
}

STUDY_COUNTS: dict[str, int] = {
    "s1": 280, "s2": 144, "s3": 64,
    "s4":  45, "s5":  96, "s6": 64,
    "s7":  72, "s8":  40,
}


# ─── Execution helpers ────────────────────────────────────────────────────────

def _build_payload(plan: dict) -> dict:
    """Strip private/meta keys and return a clean JSON payload for the runner."""
    skip = {"_bc", "_mat", "_geo", "_study"}
    return {k: v for k, v in plan.items() if k not in skip}


def run_one(plan: dict, kg, dry_run: bool = False) -> str:
    """Execute a single simulation plan.  Returns 'ok' | 'skip' | 'fail' | 'dry'."""
    run_id = plan["run_id"]

    if not dry_run and run_exists(run_id):
        return "skip"

    if dry_run:
        a = _alpha(plan)
        L = plan.get("_geo", "?")
        print(f"    DRY  {run_id}  α={a:.2e} m²/s  t_end={plan['t_end']}s  "
              f"dt={plan['dt']}s  geo={L}")
        return "dry"

    payload = _build_payload(plan)
    payload["output_dir"] = "/workspace/results"

    try:
        create_run(run_id, plan)
    except Exception:
        pass

    t0 = time.time()
    try:
        resp = requests.post(FENICS_URL, json=payload, timeout=900)
        resp.raise_for_status()
        result = resp.json()
    except Exception as exc:
        print(f"    HTTP-ERROR: {exc}", flush=True)
        try:
            mark_run_finished(run_id, {}, RunStatus.FAILED)
        except Exception:
            pass
        return "fail"

    elapsed = time.time() - t0

    if result.get("status") != "success":
        err = str(result.get("error_message", result.get("error", "unknown")))[:120]
        print(f"    FAILED ({elapsed:.1f}s): {err}", flush=True)
        try:
            mark_run_finished(run_id, result, RunStatus.FAILED)
        except Exception:
            pass
        return "fail"

    try:
        mark_run_finished(run_id, result, RunStatus.SUCCESS)
    except Exception:
        pass

    warnings: list = []
    try:
        warnings = check_config_warnings(plan)
    except Exception:
        pass

    kg_tag = "✓KG"
    try:
        kg.add_run(run_id, plan, result, warnings)
    except Exception as exc:
        kg_tag = f"✗KG({str(exc)[:30]})"

    t_max = result.get("t_max") or result.get("max_temperature", "?")
    t_min = result.get("t_min") or result.get("min_temperature", "?")
    try:
        t_str = f"T=[{float(t_min):.0f}..{float(t_max):.0f}]K"
    except Exception:
        t_str = "T=?"

    warn_tag = f"  warn={[w['code'] for w in warnings]}" if warnings else ""
    print(f"    ✓ {elapsed:.1f}s  {t_str}  {kg_tag}{warn_tag}", flush=True)
    return "ok"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full parameter sweep — 805+ heat-equation runs across 8 studies"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without running anything",
    )
    parser.add_argument(
        "--study", nargs="+", metavar="KEY",
        choices=list(STUDY_FUNCS.keys()),
        help="Run only these studies (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Stop after N simulations across selected studies (0 = no limit)",
    )
    args = parser.parse_args()

    selected = args.study or list(STUDY_FUNCS.keys())

    # ── Collect plans ────────────────────────────────────────────────────────
    all_plans: list[dict] = []
    for key in selected:
        plans = STUDY_FUNCS[key]()
        for p in plans:
            p["_study"] = key
        all_plans.extend(plans)

    if args.limit:
        all_plans = all_plans[: args.limit]

    total = len(all_plans)

    # ── Header ───────────────────────────────────────────────────────────────
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}PDE Full Parameter Sweep")
    print("=" * 72)
    for key in selected:
        n = sum(1 for p in all_plans if p["_study"] == key)
        print(f"  {key}  {STUDY_DESCRIPTIONS[key]} → {n}")
    print(f"  {'─' * 60}")
    print(f"  Total : {total} runs")
    if not args.dry_run:
        print(f"  Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()

    if args.dry_run:
        # Print a sample from each study
        for key in selected:
            study_plans = [p for p in all_plans if p["_study"] == key]
            print(f"  ── {key} sample (first 3 of {len(study_plans)}) ──")
            for p in study_plans[:3]:
                run_one(p, kg=None, dry_run=True)
        print(f"\n  Total planned: {total}")
        return

    # ── Run ──────────────────────────────────────────────────────────────────
    kg = get_kg()
    ok = fail = skip = 0
    t_wall_start = time.time()

    for i, plan in enumerate(all_plans, 1):
        key    = plan["_study"]
        run_id = plan["run_id"]
        mat    = plan.get("_mat", "?")
        bc     = plan.get("_bc",  "?")
        geo    = plan.get("_geo", "?")

        print(f"[{i:4d}/{total}] {key} | {run_id}", flush=True)
        print(f"          mat={mat}  bc={bc}  geo={geo}", flush=True)

        status = run_one(plan, kg)
        if status == "ok":
            ok += 1
        elif status == "fail":
            fail += 1
        elif status == "skip":
            skip += 1
            print("    SKIP (already in DB)", flush=True)

        # ── Progress report every 50 ─────────────────────────────────────
        if i % 50 == 0:
            elapsed_s = time.time() - t_wall_start
            rate_per_s = i / elapsed_s if elapsed_s > 0 else 0
            remaining  = (total - i) / rate_per_s if rate_per_s > 0 else 0
            eta_str    = str(timedelta(seconds=int(remaining)))
            print(
                f"\n  ── [{i}/{total}] OK={ok}  FAIL={fail}  SKIP={skip} "
                f"| {rate_per_s * 60:.1f} runs/min | ETA {eta_str} ──\n",
                flush=True,
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    total_elapsed = timedelta(seconds=int(time.time() - t_wall_start))
    print(f"\n{'=' * 72}")
    print(f"  Finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Elapsed  : {total_elapsed}")
    print(f"  OK={ok}  FAIL={fail}  SKIP={skip}  TOTAL={total}")
    try:
        stats = kg.stats()
        print(f"  KG stats : {stats}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
