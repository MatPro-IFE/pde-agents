"""
Benchmark tasks for the KG ablation study.

Each task is a natural-language simulation request that the agent system must
interpret, configure, and execute. Tasks are graded by difficulty and coverage:

  - easy:   Explicit parameters, straightforward setup
  - medium: Requires material lookup or domain knowledge
  - hard:   Ambiguous description, multiple BCs, needs KG for failure avoidance

Ground truth configs are provided so that output quality can be scored.
"""

from __future__ import annotations

ABLATION_TASKS = [
    # ── Easy: explicit parameters ────────────────────────────────────────────
    {
        "id": "E1",
        "difficulty": "easy",
        "description": (
            "Run a 2D heat equation on a unit square. "
            "Left wall at 0 K, right wall at 1 K, top and bottom insulated. "
            "Use 32x32 mesh, k=1, rho=1, cp=1, t_end=0.5, dt=0.01."
        ),
        "ground_truth": {
            "dim": 2, "nx": 32, "ny": 32,
            "k": 1.0, "rho": 1.0, "cp": 1.0,
            "t_end": 0.5, "dt": 0.01,
            "T_max_range": [0.95, 1.05],
            "T_min_range": [-0.05, 0.05],
            "should_succeed": True,
        },
    },
    {
        "id": "E2",
        "difficulty": "easy",
        "description": (
            "Simulate heat conduction in 2D with k=10 W/(mK), rho=1000 kg/m3, "
            "cp=500 J/(kgK). Left=273K, right=373K, top and bottom Neumann zero. "
            "64x64 mesh, run for 2 seconds, dt=0.05."
        ),
        "ground_truth": {
            "dim": 2, "nx": 64, "ny": 64,
            "k": 10.0, "rho": 1000.0, "cp": 500.0,
            "t_end": 2.0, "dt": 0.05,
            "T_max_range": [370, 376],
            "T_min_range": [270, 276],
            "should_succeed": True,
        },
    },
    {
        "id": "E3",
        "difficulty": "easy",
        "description": (
            "2D heat equation, all walls at T=500K, uniform initial temperature 300K. "
            "k=50, rho=7800, cp=500. 32x32, t_end=10.0, dt=0.5."
        ),
        "ground_truth": {
            "dim": 2, "nx": 32, "ny": 32,
            "k": 50.0, "rho": 7800.0, "cp": 500.0,
            "t_end": 10.0, "dt": 0.5,
            "T_max_range": [499, 501],
            "T_min_range": [290, 500],
            "should_succeed": True,
        },
    },

    # ── Medium: requires material knowledge or inference ─────────────────────
    {
        "id": "M1",
        "difficulty": "medium",
        "description": (
            "Simulate heat conduction in a steel plate (AISI 1010). "
            "Left wall at 300K, right wall at 500K. "
            "Insulated top and bottom, 64x64 mesh, run for 50 seconds."
        ),
        "ground_truth": {
            "dim": 2, "nx": 64, "ny": 64,
            "k_range": [45, 55],
            "rho_range": [7800, 7900],
            "cp_range": [480, 520],
            "T_max_range": [495, 505],
            "T_min_range": [295, 305],
            "should_succeed": True,
            "needs_kg": True,
        },
    },
    {
        "id": "M2",
        "difficulty": "medium",
        "description": (
            "Model heat transfer in a copper block. The left face is at 400K "
            "and the right face is cooled by convection (h=25 W/m2K, T_amb=293K). "
            "Top and bottom are insulated. 48x48 mesh, simulate 5 seconds."
        ),
        "ground_truth": {
            "dim": 2, "nx": 48, "ny": 48,
            "k_range": [380, 401],
            "rho_range": [8900, 8960],
            "cp_range": [380, 400],
            "has_robin_bc": True,
            "T_max_range": [395, 405],
            "should_succeed": True,
            "needs_kg": True,
        },
    },
    {
        "id": "M3",
        "difficulty": "medium",
        "description": (
            "Simulate heat diffusion in aluminium with a constant volumetric heat "
            "source of 1000 W/m3. All boundaries at 293K. 32x32 mesh, t_end=2s."
        ),
        "ground_truth": {
            "dim": 2, "nx": 32, "ny": 32,
            "k_range": [200, 240],
            "source": 1000.0,
            "T_max_range": [293, 296],
            "should_succeed": True,
            "needs_kg": True,
        },
    },

    # ── Hard: ambiguous, tricky numerics, or failure-prone ───────────────────
    {
        "id": "H1",
        "difficulty": "hard",
        "description": (
            "Simulate rapid quenching of a hot steel plate. Initial temperature "
            "is 1200K. All surfaces are convectively cooled with h=500 W/m2K and "
            "ambient 293K. Use a fine mesh (96x96) and simulate for 60 seconds."
        ),
        "ground_truth": {
            "dim": 2, "nx": 96, "ny": 96,
            "k_range": [45, 55],
            "T_max_range": [293, 1200],
            "should_succeed": True,
            "needs_kg": True,
            "numerically_tricky": True,
        },
    },
    {
        "id": "H2",
        "difficulty": "hard",
        "description": (
            "Run a 3D heat equation for a ceramic insulator. "
            "One face at 1000K, opposite face at 300K, other four faces insulated. "
            "24x24x24 mesh, run for 20 seconds."
        ),
        "ground_truth": {
            "dim": 3,
            "nx": 24, "ny": 24, "nz": 24,
            "k_range": [1, 5],
            "should_succeed": True,
            "needs_kg": True,
        },
    },
    {
        "id": "H3",
        "difficulty": "hard",
        "description": (
            "Model steady-state heat conduction with a very small time step. "
            "Material: titanium. Left=500K, right=300K. "
            "Use Crank-Nicolson (theta=0.5) with dt=0.001 and t_end=0.5. "
            "128x128 mesh."
        ),
        "ground_truth": {
            "dim": 2,
            "nx": 128, "ny": 128,
            "theta": 0.5,
            "k_range": [15, 25],
            "should_succeed": True,
            "needs_kg": True,
            "computationally_expensive": True,
        },
    },
    {
        "id": "H4",
        "difficulty": "hard",
        "description": (
            "I need to simulate heat in 'something like stainless steel 304' "
            "with mixed boundary conditions. Left heated to 800K, top cooled by "
            "convection (h=100, T_inf=300K), bottom insulated, right at 400K. "
            "Fine mesh, run for a minute of physical time."
        ),
        "ground_truth": {
            "dim": 2,
            "k_range": [14, 17],
            "rho_range": [7900, 8100],
            "has_robin_bc": True,
            "T_max_range": [795, 805],
            "should_succeed": True,
            "needs_kg": True,
        },
    },
]


def get_tasks_by_difficulty(difficulty: str | None = None) -> list[dict]:
    if difficulty is None:
        return ABLATION_TASKS
    return [t for t in ABLATION_TASKS if t["difficulty"] == difficulty]
