"""
Benchmark tasks v2 for the KG ablation study — 50 tasks total.

Design principles:
  - 50 tasks give ±14% CI at 95% confidence (vs ±37% with 7 tasks)
  - Balanced across 4 difficulty levels (12 easy, 15 medium, 13 hard, 10 novel)
  - Each task tests a distinct combination of material / BC / mesh / time scale
  - Ground truth ranges allow objective scoring without manual inspection
  - Novel tasks reference 3 fictional materials (Novidium, Cryonite, Pyrathane)
    whose properties exist ONLY in the KG

Difficulty scale:
  easy   — All parameters explicit, no material lookup needed
  medium — Requires material property knowledge (from KG or LLM memory)
  hard   — Ambiguous descriptions, mixed BCs, tricky numerics
  novel  — Fictional materials; KG is the only source of truth
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
#  EASY TASKS (12) — explicit parameters, straightforward setup
# ═══════════════════════════════════════════════════════════════════════════════

EASY_TASKS = [
    {
        "id": "E01",
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
        },
    },
    {
        "id": "E02",
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
        },
    },
    {
        "id": "E03",
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
        },
    },
    {
        "id": "E04",
        "difficulty": "easy",
        "description": (
            "2D steady-state heat conduction. k=25, left wall at 400K, "
            "right wall at 200K, top and bottom insulated. "
            "48x48 mesh, t_end=50.0, dt=0.5, rho=2000, cp=800."
        ),
        "ground_truth": {
            "dim": 2, "nx": 48, "ny": 48,
            "k": 25.0, "rho": 2000.0, "cp": 800.0,
            "T_max_range": [395, 405],
            "T_min_range": [195, 205],
        },
    },
    {
        "id": "E05",
        "difficulty": "easy",
        "description": (
            "Simulate a 2D thermal problem. Bottom wall fixed at 600K, "
            "top wall at 300K, left and right insulated. "
            "k=100, rho=8000, cp=450. 40x40 mesh, t_end=5.0, dt=0.1."
        ),
        "ground_truth": {
            "dim": 2, "nx": 40, "ny": 40,
            "k": 100.0, "rho": 8000.0, "cp": 450.0,
            "T_max_range": [595, 605],
            "T_min_range": [295, 305],
        },
    },
    {
        "id": "E06",
        "difficulty": "easy",
        "description": (
            "3D heat equation on a unit cube. k=5, rho=3000, cp=1000. "
            "Left face at 350K, right face at 250K, all other faces insulated. "
            "16x16x16 mesh, t_end=1.0, dt=0.05."
        ),
        "ground_truth": {
            "dim": 3, "nx": 16, "ny": 16, "nz": 16,
            "k": 5.0, "rho": 3000.0, "cp": 1000.0,
            "T_max_range": [345, 355],
            "T_min_range": [245, 255],
        },
    },
    {
        "id": "E07",
        "difficulty": "easy",
        "description": (
            "2D heat conduction with a uniform volumetric heat source of 500 W/m3. "
            "All boundaries at 293K. k=2, rho=1500, cp=1200. "
            "32x32 mesh, t_end=10.0, dt=0.5."
        ),
        "ground_truth": {
            "dim": 2, "nx": 32, "ny": 32,
            "k": 2.0, "rho": 1500.0, "cp": 1200.0,
            "source": 500.0,
            "T_max_range": [293, 310],
        },
    },
    {
        "id": "E08",
        "difficulty": "easy",
        "description": (
            "2D heat equation with k=0.5 (insulating material). "
            "Left=500K, right=300K, top and bottom insulated. "
            "24x24 mesh, t_end=100.0, dt=1.0, rho=500, cp=2000."
        ),
        "ground_truth": {
            "dim": 2, "nx": 24, "ny": 24,
            "k": 0.5, "rho": 500.0, "cp": 2000.0,
            "T_max_range": [495, 505],
            "T_min_range": [295, 305],
        },
    },
    {
        "id": "E09",
        "difficulty": "easy",
        "description": (
            "2D heat conduction using Crank-Nicolson (theta=0.5). "
            "k=15, rho=4000, cp=600. Left=800K, right=200K, top/bottom insulated. "
            "64x64 mesh, t_end=20.0, dt=0.2."
        ),
        "ground_truth": {
            "dim": 2, "nx": 64, "ny": 64,
            "k": 15.0, "rho": 4000.0, "cp": 600.0,
            "theta": 0.5,
            "T_max_range": [795, 805],
            "T_min_range": [195, 205],
        },
    },
    {
        "id": "E10",
        "difficulty": "easy",
        "description": (
            "2D transient heat equation. Initial temperature everywhere is 1000K. "
            "All four walls at 273K. k=30, rho=5000, cp=400. "
            "48x48 mesh, t_end=5.0, dt=0.1."
        ),
        "ground_truth": {
            "dim": 2, "nx": 48, "ny": 48,
            "k": 30.0, "rho": 5000.0, "cp": 400.0,
            "T_max_range": [273, 1000],
            "T_min_range": [270, 276],
        },
    },
    {
        "id": "E11",
        "difficulty": "easy",
        "description": (
            "3D heat equation. All six faces at 293K, initial T=500K. "
            "k=10, rho=2700, cp=900. 12x12x12 mesh, t_end=2.0, dt=0.1."
        ),
        "ground_truth": {
            "dim": 3, "nx": 12, "ny": 12, "nz": 12,
            "k": 10.0, "rho": 2700.0, "cp": 900.0,
            "T_max_range": [293, 500],
            "T_min_range": [290, 296],
        },
    },
    {
        "id": "E12",
        "difficulty": "easy",
        "description": (
            "2D heat equation with high thermal conductivity: k=400. "
            "Left=350K, right=300K, top/bottom insulated. "
            "rho=8900, cp=385. 32x32 mesh, t_end=0.01, dt=0.0005."
        ),
        "ground_truth": {
            "dim": 2, "nx": 32, "ny": 32,
            "k": 400.0, "rho": 8900.0, "cp": 385.0,
            "T_max_range": [345, 355],
            "T_min_range": [295, 305],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIUM TASKS (15) — requires material property knowledge
# ═══════════════════════════════════════════════════════════════════════════════

MEDIUM_TASKS = [
    {
        "id": "M01",
        "difficulty": "medium",
        "description": (
            "Simulate heat conduction in a steel plate (AISI 1010). "
            "Left wall at 300K, right wall at 500K. "
            "Insulated top and bottom, 64x64 mesh, run for 50 seconds."
        ),
        "ground_truth": {
            "k_range": [45, 55], "rho_range": [7800, 7900], "cp_range": [480, 520],
            "T_max_range": [495, 505], "T_min_range": [295, 305],
        },
    },
    {
        "id": "M02",
        "difficulty": "medium",
        "description": (
            "Model heat transfer in a copper block. The left face is at 400K "
            "and the right face is cooled by convection (h=25 W/m2K, T_amb=293K). "
            "Top and bottom insulated. 48x48 mesh, simulate 5 seconds."
        ),
        "ground_truth": {
            "k_range": [380, 401], "rho_range": [8900, 8960], "cp_range": [380, 400],
            "has_robin_bc": True,
            "T_max_range": [395, 405],
        },
    },
    {
        "id": "M03",
        "difficulty": "medium",
        "description": (
            "Simulate heat diffusion in aluminium with a constant volumetric heat "
            "source of 1000 W/m3. All boundaries at 293K. 32x32 mesh, t_end=2s."
        ),
        "ground_truth": {
            "k_range": [200, 240], "source": 1000.0,
            "T_max_range": [293, 296],
        },
    },
    {
        "id": "M04",
        "difficulty": "medium",
        "description": (
            "Simulate heat conduction in a brass plate. "
            "Left wall at 500K, right wall at 300K, top and bottom insulated. "
            "48x48 mesh, run for 10 seconds."
        ),
        "ground_truth": {
            "k_range": [100, 130], "rho_range": [8400, 8700], "cp_range": [370, 400],
            "T_max_range": [495, 505], "T_min_range": [295, 305],
        },
    },
    {
        "id": "M05",
        "difficulty": "medium",
        "description": (
            "Model steady-state heat transfer in a glass window pane. "
            "Interior face at 293K, exterior face at 263K. "
            "Top and bottom insulated. 32x32 mesh, t_end=300s."
        ),
        "ground_truth": {
            "k_range": [0.8, 1.2], "rho_range": [2400, 2600], "cp_range": [750, 850],
            "T_max_range": [290, 296], "T_min_range": [260, 266],
        },
    },
    {
        "id": "M06",
        "difficulty": "medium",
        "description": (
            "Simulate transient heat conduction in a nickel block. "
            "Left face at 800K, right face at 300K. "
            "Insulated top and bottom. 64x64, run for 30 seconds."
        ),
        "ground_truth": {
            "k_range": [88, 92], "rho_range": [8850, 8950], "cp_range": [440, 465],
            "T_max_range": [795, 805], "T_min_range": [295, 305],
        },
    },
    {
        "id": "M07",
        "difficulty": "medium",
        "description": (
            "Heat transfer in a concrete wall. Left face at 350K "
            "(fire side), right face at 293K (room side). "
            "Top and bottom insulated. 32x32 mesh, t_end=600 seconds."
        ),
        "ground_truth": {
            "k_range": [1.0, 1.8], "rho_range": [2200, 2500], "cp_range": [850, 1000],
            "T_max_range": [345, 355], "T_min_range": [290, 296],
        },
    },
    {
        "id": "M08",
        "difficulty": "medium",
        "description": (
            "Model heat conduction in a platinum plate. "
            "Left=600K, right=400K, insulated top/bottom. "
            "48x48 mesh, t_end=20s."
        ),
        "ground_truth": {
            "k_range": [68, 75], "rho_range": [21000, 21600], "cp_range": [130, 140],
            "T_max_range": [595, 605], "T_min_range": [395, 405],
        },
    },
    {
        "id": "M09",
        "difficulty": "medium",
        "description": (
            "Simulate heat transfer through a cast iron block. "
            "Bottom at 700K, top at 300K, left and right insulated. "
            "64x64 mesh, run for 100 seconds."
        ),
        "ground_truth": {
            "k_range": [45, 55], "rho_range": [7100, 7400], "cp_range": [460, 540],
            "T_max_range": [695, 705], "T_min_range": [295, 305],
        },
    },
    {
        "id": "M10",
        "difficulty": "medium",
        "description": (
            "Heat diffusion in a tungsten plate. "
            "Left=1500K, right=500K, insulated top and bottom. "
            "32x32 mesh, run for 5 seconds."
        ),
        "ground_truth": {
            "k_range": [160, 180], "rho_range": [19200, 19400], "cp_range": [130, 140],
            "T_max_range": [1495, 1505], "T_min_range": [495, 505],
        },
    },
    {
        "id": "M11",
        "difficulty": "medium",
        "description": (
            "Model heat transfer in a polycarbonate plastic sheet. "
            "Left face at 380K, right face at 293K, insulated top/bottom. "
            "32x32 mesh, t_end=200s."
        ),
        "ground_truth": {
            "k_range": [0.19, 0.22], "rho_range": [1150, 1250], "cp_range": [1100, 1300],
            "T_max_range": [375, 385], "T_min_range": [290, 296],
        },
    },
    {
        "id": "M12",
        "difficulty": "medium",
        "description": (
            "Simulate heat conduction in a lead block. "
            "Left=450K, right=293K, top and bottom insulated. "
            "48x48, t_end=15 seconds."
        ),
        "ground_truth": {
            "k_range": [33, 37], "rho_range": [11300, 11400], "cp_range": [127, 132],
            "T_max_range": [445, 455], "T_min_range": [290, 296],
        },
    },
    {
        "id": "M13",
        "difficulty": "medium",
        "description": (
            "Simulate thermal conduction in a silicon wafer. "
            "Left edge at 400K, right edge at 300K, insulated top/bottom. "
            "64x64, run for 0.5 seconds."
        ),
        "ground_truth": {
            "k_range": [140, 160], "rho_range": [2300, 2340], "cp_range": [700, 720],
            "T_max_range": [395, 405], "T_min_range": [295, 305],
        },
    },
    {
        "id": "M14",
        "difficulty": "medium",
        "description": (
            "Heat transfer in a zinc plate. Left at 500K, right at 293K, "
            "top and bottom insulated. 32x32 mesh, run for 10 seconds."
        ),
        "ground_truth": {
            "k_range": [110, 120], "rho_range": [7100, 7200], "cp_range": [385, 400],
            "T_max_range": [495, 505], "T_min_range": [290, 296],
        },
    },
    {
        "id": "M15",
        "difficulty": "medium",
        "description": (
            "Simulate heat conduction in stainless steel 316 with "
            "convective cooling on the right face (h=50, T_amb=293K). "
            "Left face at 500K. Top/bottom insulated. 48x48, t_end=60s."
        ),
        "ground_truth": {
            "k_range": [14, 17], "rho_range": [7900, 8100], "cp_range": [490, 510],
            "has_robin_bc": True,
            "T_max_range": [495, 505],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HARD TASKS (13) — ambiguous, tricky numerics, mixed BCs
# ═══════════════════════════════════════════════════════════════════════════════

HARD_TASKS = [
    {
        "id": "H01",
        "difficulty": "hard",
        "description": (
            "Simulate rapid quenching of a hot steel plate. Initial temperature "
            "is 1200K. All surfaces are convectively cooled with h=500 W/m2K and "
            "ambient 293K. Use a fine mesh (96x96) and simulate for 60 seconds."
        ),
        "ground_truth": {
            "k_range": [45, 55],
            "T_max_range": [293, 1200],
            "numerically_tricky": True,
        },
    },
    {
        "id": "H02",
        "difficulty": "hard",
        "description": (
            "Run a 3D heat equation for a ceramic insulator. "
            "One face at 1000K, opposite face at 300K, other four faces insulated. "
            "24x24x24 mesh, run for 20 seconds."
        ),
        "ground_truth": {
            "dim": 3,
            "k_range": [1, 5],
        },
    },
    {
        "id": "H03",
        "difficulty": "hard",
        "description": (
            "Model steady-state heat conduction with a very small time step. "
            "Material: titanium. Left=500K, right=300K. "
            "Use Crank-Nicolson (theta=0.5) with dt=0.001 and t_end=0.5. "
            "128x128 mesh."
        ),
        "ground_truth": {
            "k_range": [15, 25],
            "theta": 0.5,
            "computationally_expensive": True,
        },
    },
    {
        "id": "H04",
        "difficulty": "hard",
        "description": (
            "I need to simulate heat in 'something like stainless steel 304' "
            "with mixed boundary conditions. Left heated to 800K, top cooled by "
            "convection (h=100, T_inf=300K), bottom insulated, right at 400K. "
            "Fine mesh, run for a minute of physical time."
        ),
        "ground_truth": {
            "k_range": [14, 17], "rho_range": [7900, 8100],
            "has_robin_bc": True,
            "T_max_range": [795, 805],
        },
    },
    {
        "id": "H05",
        "difficulty": "hard",
        "description": (
            "Simulate thermal shock: a copper plate initially at 293K, "
            "suddenly exposed to 1200K on the left face. Right face at 293K. "
            "Top and bottom insulated. Very fine mesh (128x128), "
            "simulate 0.1 seconds with small time steps."
        ),
        "ground_truth": {
            "k_range": [380, 401], "rho_range": [8900, 8960],
            "T_max_range": [1195, 1205],
            "T_min_range": [290, 296],
            "numerically_tricky": True,
        },
    },
    {
        "id": "H06",
        "difficulty": "hard",
        "description": (
            "3D heat transfer in an aluminium heat sink. "
            "Bottom face at 373K (chip contact), top face convective cooling "
            "(h=200 W/m2K, T_amb=293K). All side faces insulated. "
            "20x20x20 mesh, run for 5 seconds."
        ),
        "ground_truth": {
            "dim": 3,
            "k_range": [200, 240], "rho_range": [2680, 2720],
            "has_robin_bc": True,
        },
    },
    {
        "id": "H07",
        "difficulty": "hard",
        "description": (
            "Model heat diffusion in a poorly described material — 'some kind of "
            "iron alloy, maybe cast iron'. Left wall at 600K, right wall at 293K, "
            "top has mild convection (h=10, T_inf=293K), bottom insulated. "
            "64x64, run for 200 seconds."
        ),
        "ground_truth": {
            "k_range": [40, 55], "rho_range": [7000, 7500],
            "has_robin_bc": True,
            "T_max_range": [595, 605],
        },
    },
    {
        "id": "H08",
        "difficulty": "hard",
        "description": (
            "Simulate a thin concrete slab heated by a fire. Left face exposed to "
            "1100K, right face at ambient 293K. Top and bottom insulated. "
            "The concrete is dense (about 2400 kg/m3). Use a fine mesh. "
            "Run for 30 minutes of physical time."
        ),
        "ground_truth": {
            "k_range": [1.0, 1.8], "rho_range": [2200, 2500],
            "T_max_range": [1095, 1105], "T_min_range": [290, 296],
        },
    },
    {
        "id": "H09",
        "difficulty": "hard",
        "description": (
            "3D heat conduction in a titanium aerospace component. "
            "One face at 500K, opposite face at 300K, two side faces "
            "convectively cooled (h=75, T_inf=293K), remaining two insulated. "
            "16x16x16 mesh, simulate for 30 seconds."
        ),
        "ground_truth": {
            "dim": 3,
            "k_range": [15, 25], "rho_range": [4400, 4550],
            "has_robin_bc": True,
        },
    },
    {
        "id": "H10",
        "difficulty": "hard",
        "description": (
            "Simulate heat in a nickel superalloy (Inconel 718). "
            "Left wall at 900K, right wall cooled by convection (h=150, T_amb=293K). "
            "Top at 700K, bottom insulated. 96x96, run for 120 seconds."
        ),
        "ground_truth": {
            "k_range": [11, 14], "rho_range": [8150, 8250],
            "has_robin_bc": True,
            "T_max_range": [895, 905],
        },
    },
    {
        "id": "H11",
        "difficulty": "hard",
        "description": (
            "Model the heating of a large granite block. It starts at 293K and "
            "the left face is heated to 800K. Right at 293K. "
            "Top and bottom insulated. It's a very slow process — simulate "
            "for 1 hour of physical time. 48x48 mesh."
        ),
        "ground_truth": {
            "k_range": [2.5, 3.5], "rho_range": [2600, 2700], "cp_range": [790, 830],
            "T_max_range": [795, 805], "T_min_range": [290, 296],
        },
    },
    {
        "id": "H12",
        "difficulty": "hard",
        "description": (
            "Simulate heat transfer in a silver plate with all four boundaries "
            "at different temperatures: left=500K, right=300K, top=400K, bottom=350K. "
            "48x48 mesh, run for 2 seconds."
        ),
        "ground_truth": {
            "k_range": [420, 430], "rho_range": [10400, 10600], "cp_range": [232, 240],
            "T_max_range": [495, 505], "T_min_range": [295, 355],
        },
    },
    {
        "id": "H13",
        "difficulty": "hard",
        "description": (
            "Simulate heat conduction in a carbon fiber composite. "
            "This is an anisotropic material but use an effective thermal "
            "conductivity. Left face 500K, right face 293K. Top and bottom "
            "insulated. 64x64, run for 20 seconds."
        ),
        "ground_truth": {
            "k_range": [5, 25], "rho_range": [1500, 1800], "cp_range": [700, 1000],
            "T_max_range": [495, 505], "T_min_range": [290, 296],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  NOVEL TASKS (10) — fictional materials, KG is the only source of truth
# ═══════════════════════════════════════════════════════════════════════════════

NOVEL_TASKS = [
    # ── Novidium (k≈73, rho≈5420, cp≈612) ────────────────────────────────────
    {
        "id": "N01",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate 2D steady-state heat conduction in a Novidium plate. "
            "Left wall at 400K, right wall at 300K, top and bottom insulated. "
            "48x48 mesh. Look up Novidium properties from the knowledge graph."
        ),
        "ground_truth": {
            "k_range": [70.0, 76.0], "rho_range": [5400.0, 5440.0],
            "cp_range": [600.0, 625.0],
            "T_max_range": [395, 405], "T_min_range": [295, 305],
        },
    },
    {
        "id": "N02",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Model transient heat diffusion in a Novidium block (2D). "
            "All four boundaries at 293K, initial temperature 600K. "
            "32x32 mesh, simulate for 200 seconds with dt=1.0. "
            "Use the material properties of Novidium."
        ),
        "ground_truth": {
            "k_range": [70.0, 76.0], "rho_range": [5400.0, 5440.0],
            "cp_range": [600.0, 625.0],
            "T_max_range": [293, 600], "T_min_range": [290, 296],
        },
    },
    {
        "id": "N03",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate Novidium with mixed BCs. Left 700K, right convection "
            "(h=50, T_amb=293K). Top/bottom insulated. 64x64, 150 seconds. "
            "Use Novidium thermal properties."
        ),
        "ground_truth": {
            "k_range": [70.0, 76.0], "rho_range": [5400.0, 5440.0],
            "cp_range": [600.0, 625.0],
            "has_robin_bc": True,
            "T_max_range": [695, 705],
        },
    },
    {
        "id": "N04",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Run a 3D heat simulation in Novidium. Left face 500K, right face 293K, "
            "all other faces insulated. 16x16x16 mesh, t_end=50s. "
            "Look up Novidium properties."
        ),
        "ground_truth": {
            "dim": 3,
            "k_range": [70.0, 76.0], "rho_range": [5400.0, 5440.0],
            "cp_range": [600.0, 625.0],
            "T_max_range": [495, 505], "T_min_range": [290, 296],
        },
    },
    # ── Cryonite (k≈0.42, rho≈1180, cp≈1940) ────────────────────────────────
    {
        "id": "N05",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate 2D steady-state heat conduction in a Cryonite insulation "
            "panel. Left wall at 350K, right wall at 280K, top and bottom "
            "insulated. 48x48 mesh. Look up Cryonite properties."
        ),
        "ground_truth": {
            "k_range": [0.38, 0.46], "rho_range": [1150.0, 1210.0],
            "cp_range": [1900.0, 1980.0],
            "T_max_range": [345, 355], "T_min_range": [275, 285],
        },
    },
    {
        "id": "N06",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Model heat transfer through a Cryonite wall with convective "
            "cooling on the right face (h=15, T_amb=250K). Left face at 400K, "
            "top/bottom insulated. 32x32 mesh, t_end=500s. Use Cryonite."
        ),
        "ground_truth": {
            "k_range": [0.38, 0.46], "rho_range": [1150.0, 1210.0],
            "cp_range": [1900.0, 1980.0],
            "has_robin_bc": True,
            "T_max_range": [395, 405],
        },
    },
    {
        "id": "N07",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate transient cooling of a Cryonite block. "
            "Initial temperature 400K, all boundaries at 250K. "
            "48x48 mesh, t_end=1000 seconds. Use Cryonite properties."
        ),
        "ground_truth": {
            "k_range": [0.38, 0.46], "rho_range": [1150.0, 1210.0],
            "cp_range": [1900.0, 1980.0],
            "T_max_range": [250, 400], "T_min_range": [247, 253],
        },
    },
    # ── Pyrathane (k≈312, rho≈3850, cp≈278) ──────────────────────────────────
    {
        "id": "N08",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate 2D steady-state heat conduction in a Pyrathane crucible wall. "
            "Left at 1500K, right at 400K, top/bottom insulated. "
            "48x48 mesh. Look up Pyrathane properties."
        ),
        "ground_truth": {
            "k_range": [305.0, 320.0], "rho_range": [3800.0, 3900.0],
            "cp_range": [270.0, 286.0],
            "T_max_range": [1495, 1505], "T_min_range": [395, 405],
        },
    },
    {
        "id": "N09",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate transient heat diffusion in a Pyrathane component. "
            "All boundaries at 400K, initial temperature 2000K. "
            "32x32 mesh, t_end=15 seconds, dt=0.1. Use Pyrathane properties."
        ),
        "ground_truth": {
            "k_range": [305.0, 320.0], "rho_range": [3800.0, 3900.0],
            "cp_range": [270.0, 286.0],
            "T_max_range": [400, 2000], "T_min_range": [395, 405],
        },
    },
    {
        "id": "N10",
        "difficulty": "novel",
        "material_novelty": "novel",
        "description": (
            "Simulate Pyrathane with convective cooling. Left face at 1000K, "
            "right face convection (h=200, T_amb=400K). Top/bottom insulated. "
            "64x64, run for 10 seconds. Use Pyrathane properties."
        ),
        "ground_truth": {
            "k_range": [305.0, 320.0], "rho_range": [3800.0, 3900.0],
            "cp_range": [270.0, 286.0],
            "has_robin_bc": True,
            "T_max_range": [995, 1005],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Combined exports
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TASKS_V2 = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS + NOVEL_TASKS

assert len(ALL_TASKS_V2) == 50, f"Expected 50 tasks, got {len(ALL_TASKS_V2)}"
assert len({t["id"] for t in ALL_TASKS_V2}) == 50, "Duplicate task IDs"


def get_tasks_by_difficulty(difficulty: str | None = None) -> list[dict]:
    if difficulty is None:
        return ALL_TASKS_V2
    return [t for t in ALL_TASKS_V2 if t["difficulty"] == difficulty]


def get_novel_tasks() -> list[dict]:
    return NOVEL_TASKS


def get_standard_tasks() -> list[dict]:
    """Non-novel tasks (easy + medium + hard)."""
    return EASY_TASKS + MEDIUM_TASKS + HARD_TASKS


def get_all_tasks() -> list[dict]:
    return ALL_TASKS_V2


# Quick validation
if __name__ == "__main__":
    by_diff = {}
    for t in ALL_TASKS_V2:
        by_diff.setdefault(t["difficulty"], []).append(t["id"])
    for d, ids in by_diff.items():
        print(f"  {d:<8s}: {len(ids)} tasks — {', '.join(ids)}")
    print(f"\n  Total: {len(ALL_TASKS_V2)} tasks")
