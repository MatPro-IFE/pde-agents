"""
Static physical knowledge seeder for the Simulation Knowledge Graph.

Populates Neo4j with:
  - Engineering materials (thermal properties)
  - Known PDE failure patterns (KnownIssue nodes)
  - Physical stability rules

Run once at startup via: SimulationKnowledgeGraph.seed_if_empty()
"""

from __future__ import annotations

# ─── Engineering materials ────────────────────────────────────────────────────
# (name, k W/m·K, rho kg/m³, cp J/kg·K, description)

MATERIALS: list[dict] = [
    {
        "name": "Steel (carbon)",
        "k": 50.0, "rho": 7800.0, "cp": 500.0,
        "alpha": 50.0 / (7800.0 * 500.0),
        "description": "Common structural steel. Good baseline for mechanical FEM.",
        "typical_uses": "plates, beams, pressure vessels",
        "k_range": (40.0, 60.0),
    },
    {
        "name": "Stainless Steel (316)",
        "k": 16.0, "rho": 8000.0, "cp": 500.0,
        "alpha": 16.0 / (8000.0 * 500.0),
        "description": "Low thermal conductivity — thermal gradients build up quickly.",
        "typical_uses": "food processing, chemical reactors, marine",
        "k_range": (14.0, 18.0),
    },
    {
        "name": "Aluminium (6061)",
        "k": 200.0, "rho": 2700.0, "cp": 900.0,
        "alpha": 200.0 / (2700.0 * 900.0),
        "description": "High conductivity — temperature distributes rapidly. Needs fine time stepping to resolve transients.",
        "typical_uses": "heat sinks, aerospace structures, electronics",
        "k_range": (160.0, 240.0),
    },
    {
        "name": "Copper",
        "k": 385.0, "rho": 8960.0, "cp": 385.0,
        "alpha": 385.0 / (8960.0 * 385.0),
        "description": "Very high conductivity. Transients resolve extremely quickly.",
        "typical_uses": "heat exchangers, electrical conductors, cooling fins",
        "k_range": (370.0, 400.0),
    },
    {
        "name": "Titanium (Ti-6Al-4V)",
        "k": 6.7, "rho": 4430.0, "cp": 526.0,
        "alpha": 6.7 / (4430.0 * 526.0),
        "description": "Very low conductivity for a metal. High thermal gradients expected.",
        "typical_uses": "aerospace, biomedical implants, high-temperature applications",
        "k_range": (5.0, 8.0),
    },
    {
        "name": "Silicon",
        "k": 150.0, "rho": 2330.0, "cp": 700.0,
        "alpha": 150.0 / (2330.0 * 700.0),
        "description": "High conductivity semiconductor. Important for chip thermal management.",
        "typical_uses": "microelectronics, solar cells, MEMS",
        "k_range": (100.0, 160.0),
    },
    {
        "name": "Concrete",
        "k": 1.7, "rho": 2300.0, "cp": 880.0,
        "alpha": 1.7 / (2300.0 * 880.0),
        "description": "Low conductivity, high density. Very slow thermal response — needs long t_end.",
        "typical_uses": "buildings, dams, nuclear shielding",
        "k_range": (0.8, 2.5),
    },
    {
        "name": "Water",
        "k": 0.6, "rho": 1000.0, "cp": 4182.0,
        "alpha": 0.6 / (1000.0 * 4182.0),
        "description": "Very low diffusivity — purely conductive model ignores convection.",
        "typical_uses": "baseline fluid thermal analysis (no convection)",
        "k_range": (0.55, 0.65),
    },
    {
        "name": "Air (approx.)",
        "k": 0.026, "rho": 1.2, "cp": 1005.0,
        "alpha": 0.026 / (1.2 * 1005.0),
        "description": "Very low conductivity gas. Purely conductive air model (no convection).",
        "typical_uses": "enclosure gas, insulation layer",
        "k_range": (0.024, 0.030),
    },
    {
        "name": "Glass (borosilicate)",
        "k": 1.0, "rho": 2230.0, "cp": 830.0,
        "alpha": 1.0 / (2230.0 * 830.0),
        "description": "Low conductivity ceramic. Prone to thermal shock.",
        "typical_uses": "optical components, laboratory glassware, oven windows",
        "k_range": (0.8, 1.2),
    },
    {
        "name": "Novidium",
        "k": 73.0, "rho": 5420.0, "cp": 612.0,
        "alpha": 73.0 / (5420.0 * 612.0),
        "description": (
            "Novidium is a novel ceramic-metallic composite with moderate thermal "
            "conductivity (73 W/m·K) and an unusually high density (5420 kg/m³). "
            "It exhibits a non-linear thermal response above 850 K where conductivity "
            "drops sharply to ~45 W/m·K due to micro-cracking at grain boundaries. "
            "Below 200 K it becomes a near-perfect insulator (k ≈ 0.5 W/m·K). "
            "Safe operating range: 200–850 K. "
            "Melting point: 2150 K. Phase transition at 1680 K."
        ),
        "typical_uses": "thermal barrier coatings, high-temperature reactor liners, "
                        "aerospace re-entry shielding, fusion blanket components",
        "k_range": (70.0, 76.0),
    },
]


# ─── Known failure patterns ────────────────────────────────────────────────────

KNOWN_ISSUES: list[dict] = [
    {
        "code": "GIBBS_OVERSHOOT",
        "severity": "high",
        "condition": "u_init far from Dirichlet BC values (ΔT > 100 K)",
        "description": (
            "When the initial temperature field is inconsistent with the boundary conditions, "
            "the first few time steps produce unphysical temperature spikes (overshoot above "
            "the maximum BC value or below zero). This is a numerical artifact, not a physical result."
        ),
        "recommendation": "Set u_init equal to or close to the minimum Dirichlet BC value.",
        "observed_in": "heat_equation, any transient PDE with Dirichlet BCs",
    },
    {
        "code": "EXPLICIT_INSTABILITY",
        "severity": "high",
        "condition": "theta < 0.5 AND dt > h²/(2α)",
        "description": (
            "Explicit and near-explicit time schemes (theta < 0.5) are only conditionally stable. "
            "When the time step exceeds the diffusion stability limit, the solution oscillates "
            "and blows up exponentially."
        ),
        "recommendation": "Use theta ≥ 0.5 (Crank-Nicolson or Backward Euler).",
        "observed_in": "heat_equation with explicit Euler (theta=0)",
    },
    {
        "code": "MESH_TOO_COARSE",
        "severity": "medium",
        "condition": "nx < 10 or ny < 10 in 2D; < 8 in 3D",
        "description": (
            "A coarse mesh introduces significant spatial discretization error. "
            "The solution may converge to the wrong values, especially near boundaries "
            "where gradients are steep."
        ),
        "recommendation": "Run a mesh refinement study: compare nx=20, 40, 80 and check if T_max changes by < 1%.",
        "observed_in": "Any FEM simulation",
    },
    {
        "code": "NO_STEADY_STATE_REACHED",
        "severity": "medium",
        "condition": "L2 norm still decreasing significantly at t_end",
        "description": (
            "The simulation ended before the solution reached steady state. "
            "The final temperature field reflects a transient state, not the long-term behavior."
        ),
        "recommendation": "Increase t_end. Steady-state time scale ≈ L²/α (L=domain size, α=diffusivity).",
        "observed_in": "heat_equation with short t_end",
    },
    {
        "code": "PURE_NEUMANN_ILL_POSED",
        "severity": "high",
        "condition": "All boundary conditions are Neumann (no Dirichlet BC)",
        "description": (
            "A pure Neumann problem has no unique temperature solution — "
            "you can add any constant to the solution and it still satisfies the equations. "
            "The linear system is singular."
        ),
        "recommendation": "Add at least one Dirichlet (temperature) boundary condition.",
        "observed_in": "heat_equation with only flux BCs",
    },
    {
        "code": "NEGATIVE_TEMPERATURES",
        "severity": "high",
        "condition": "T_min < 0 K in the solution",
        "description": (
            "Temperatures below absolute zero are physically impossible and indicate either: "
            "(1) inconsistent initial condition (Gibbs overshoot), "
            "(2) an unstable time scheme, or "
            "(3) incorrect boundary condition values."
        ),
        "recommendation": "Check u_init consistency with BCs, verify theta ≥ 0.5, inspect BC values.",
        "observed_in": "heat_equation with inconsistent IC or explicit instability",
    },
]
