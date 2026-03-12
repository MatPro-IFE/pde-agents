"""
Physics reference knowledge for the PDE Agents knowledge graph.

Reference nodes encode curated, source-cited facts about:
  - Material properties (temperature-dependent conductivity, phase transitions)
  - Boundary condition engineering practice (realistic h coefficients,
    typical heat flux values, when each BC type is physically appropriate)
  - Solver guidance (mesh resolution rules, time-step stability criteria,
    element degree trade-offs)
  - Domain considerations (size-dependent effects, multi-scale cautions)

Schema addition
───────────────
  (:Reference {
      ref_id:    str   — unique identifier
      type:      str   — "material_property" | "bc_practice" | "solver_guidance"
                         | "domain_physics"
      subject:   str   — the concept being described
      text:      str   — human + LLM readable fact / warning / recommendation
      source:    str   — citation (NIST, ASHRAE, textbook, etc.)
      url:       str   — direct link to the cited document (empty string if none)
      tags:      list  — searchable keywords
  })

Relationships added
───────────────────
  (:Material)-[:HAS_REFERENCE]->(:Reference)   — material-specific facts
  (:BCConfig)-[:HAS_REFERENCE]->(:Reference)   — BC-pattern-specific guidance
"""

from __future__ import annotations

# ─── Material property references ─────────────────────────────────────────────

MATERIAL_REFERENCES: list[dict] = [
    # ── Copper ────────────────────────────────────────────────────────────────
    {
        "ref_id":  "mat_copper_k_temp",
        "type":    "material_property",
        "subject": "copper thermal conductivity temperature dependence",
        "text": (
            "Copper k drops from ~401 W/(m·K) at 20°C to ~391 W/(m·K) at 100°C "
            "and ~377 W/(m·K) at 300°C. For precision above 200°C use a "
            "temperature-dependent k(T) = 401 - 0.067*(T-20) W/(m·K) fit. "
            "Constant-k simulations at elevated temperature over-predict heat flux."
        ),
        "source": "NIST Thermophysical Properties of Matter (TPRC vol. 2)",
        "url":    "https://webbook.nist.gov/chemistry/fluid/",
        "tags":   ["copper", "temperature-dependent", "conductivity", "high_conductor"],
        "material_names": ["copper"],
    },
    {
        "ref_id":  "mat_copper_phase",
        "type":    "material_property",
        "subject": "copper melting point",
        "text": (
            "Copper melts at 1085°C. If T_max in simulation approaches 900°C "
            "or above, latent-heat and phase-change effects must be accounted for; "
            "a standard linear heat equation is no longer valid near melt."
        ),
        "source": "ASM Handbook vol. 2: Properties and Selection — Nonferrous Alloys",
        "url":    "https://www.asminternational.org/asm-handbook/",
        "tags":   ["copper", "phase_change", "melting", "validity_limit"],
        "material_names": ["copper"],
    },
    # ── Steel ─────────────────────────────────────────────────────────────────
    {
        "ref_id":  "mat_steel_k_temp",
        "type":    "material_property",
        "subject": "steel thermal conductivity temperature dependence",
        "text": (
            "Carbon steel k decreases from ~50 W/(m·K) at 20°C to ~36 W/(m·K) "
            "at 700°C (roughly linear: k ≈ 50 - 0.02*(T-20)).  Above 700°C the "
            "austenite transformation further changes properties. For high-temperature "
            "furnace / fire-resistance simulations, a constant k=50 overestimates "
            "conductivity by up to 28%."
        ),
        "source": "EN 1993-1-2 Eurocode 3: Design of steel structures — Fire resistance",
        "url":    "https://eurocodes.jrc.ec.europa.eu/EN-Eurocodes/eurocode-3-design-steel-structures",
        "tags":   ["steel", "temperature-dependent", "conductivity", "high_temperature"],
        "material_names": ["steel", "carbon_steel"],
    },
    {
        "ref_id":  "mat_steel_cp_temp",
        "type":    "material_property",
        "subject": "steel specific heat capacity temperature dependence",
        "text": (
            "Steel cp rises from ~480 J/(kg·K) at 20°C, peaks around 780°C "
            "at the Curie point (cp ≈ 5000 J/(kg·K) — a sharp spike due to "
            "magnetic transition), then drops to ~600 J/(kg·K) above 800°C. "
            "Simulations near 700–800°C must account for this anomaly; "
            "constant cp=500 causes significant error in that temperature range."
        ),
        "source": "EN 1993-1-2 Eurocode 3, clause 3.4.1.2",
        "url":    "https://eurocodes.jrc.ec.europa.eu/EN-Eurocodes/eurocode-3-design-steel-structures",
        "tags":   ["steel", "specific_heat", "curie_point", "high_temperature"],
        "material_names": ["steel"],
    },
    # ── Aluminium ─────────────────────────────────────────────────────────────
    {
        "ref_id":  "mat_aluminium_k_temp",
        "type":    "material_property",
        "subject": "aluminium thermal conductivity temperature dependence",
        "text": (
            "6061-T6 aluminium: k ≈ 167 W/(m·K) at 25°C, 170 W/(m·K) at 100°C, "
            "176 W/(m·K) at 200°C. Conductivity is nearly constant below 200°C "
            "(<5% variation), so constant-k assumption is valid for most "
            "structural thermal analyses."
        ),
        "source": "NIST TPRC vol. 2 / ASM Handbook vol. 2",
        "url":    "https://webbook.nist.gov/chemistry/fluid/",
        "tags":   ["aluminium", "temperature-dependent", "conductivity", "high_conductor"],
        "material_names": ["aluminium", "aluminum"],
    },
    # ── Concrete ──────────────────────────────────────────────────────────────
    {
        "ref_id":  "mat_concrete_k_moisture",
        "type":    "material_property",
        "subject": "concrete thermal conductivity moisture dependence",
        "text": (
            "Dry concrete k ≈ 0.8–1.0 W/(m·K); saturated concrete k ≈ 1.4–2.0 W/(m·K). "
            "Moisture content can double the effective conductivity. For civil/structural "
            "applications specify whether dry or saturated conditions apply. "
            "Standard value k=1.0 W/(m·K) assumes dry concrete."
        ),
        "source": "ISO 10456:2007 Building materials and products — thermal properties",
        "url":    "https://www.iso.org/standard/40967.html",
        "tags":   ["concrete", "moisture", "effective_conductivity", "low_conductor"],
        "material_names": ["concrete"],
    },
    # ── Water ─────────────────────────────────────────────────────────────────
    {
        "ref_id":  "mat_water_convection",
        "type":    "material_property",
        "subject": "water natural convection in FEM models",
        "text": (
            "k_water ≈ 0.6 W/(m·K) applies only to pure conduction. In liquid "
            "water domains, natural convection (Rayleigh number Ra = gβΔTL³/να) "
            "enhances heat transfer by 10–1000×. A pure heat-conduction FEM "
            "model with k=0.6 dramatically under-predicts heat transfer in any "
            "water pool or cooling channel. Use effective k or add Boussinesq flow."
        ),
        "source": "Incropera et al., Fundamentals of Heat and Mass Transfer, 7th ed.",
        "url":    "https://www.wiley.com/en-us/Fundamentals+of+Heat+and+Mass+Transfer%2C+7th+Edition-p-9780470501979",
        "tags":   ["water", "natural_convection", "validity_limit", "thermal_insulator"],
        "material_names": ["water"],
    },
    # ── Silicon ───────────────────────────────────────────────────────────────
    {
        "ref_id":  "mat_silicon_k_temp",
        "type":    "material_property",
        "subject": "silicon thermal conductivity temperature dependence",
        "text": (
            "Silicon k drops steeply with temperature: 150 W/(m·K) at 25°C, "
            "100 W/(m·K) at 100°C, 55 W/(m·K) at 200°C, 30 W/(m·K) at 400°C. "
            "This 5× reduction over typical chip operating range is critical for "
            "electronics thermal simulations. Constant-k = 150 is only valid near "
            "ambient temperature."
        ),
        "source": "Glassbrenner & Slack, Physical Review 134(4A):A1058, 1964",
        "url":    "https://journals.aps.org/pr/abstract/10.1103/PhysRev.134.A1058",
        "tags":   ["silicon", "temperature-dependent", "conductivity", "microelectronics"],
        "material_names": ["silicon"],
    },
]


# ─── Boundary condition practice references ───────────────────────────────────

BC_REFERENCES: list[dict] = [
    {
        "ref_id":  "bc_robin_h_natural_air",
        "type":    "bc_practice",
        "subject": "natural convection h coefficient in air",
        "text": (
            "Natural convection in air: h = 5–25 W/(m²·K). Use ~5–10 for vertical "
            "surfaces in calm air, ~15–25 for heated horizontal surfaces. "
            "Values outside this range indicate forced convection or poor boundary "
            "condition setup. A Robin BC with h < 2 or h > 100 in air is suspect."
        ),
        "source": "Bergman et al., Fundamentals of Heat and Mass Transfer, 7th ed., ch. 9",
        "url":    "https://www.wiley.com/en-us/Fundamentals+of+Heat+and+Mass+Transfer%2C+7th+Edition-p-9780470501979",
        "tags":   ["robin", "convection", "air", "h_coefficient", "natural_convection"],
        "bc_patterns": ["robin", "dirichlet+robin", "neumann+robin", "dirichlet+neumann+robin"],
    },
    {
        "ref_id":  "bc_robin_h_forced_air",
        "type":    "bc_practice",
        "subject": "forced convection h coefficient in air",
        "text": (
            "Forced air convection (fans, wind): h = 25–250 W/(m²·K). "
            "Low-speed fan cooling (~2 m/s): h ≈ 25–50. High-speed airflow "
            "(electronics cooling, 5–10 m/s): h ≈ 50–150. Wind at 10 m/s: "
            "h ≈ 50–100 on a flat plate. Automotive under-hood: h ≈ 100–250."
        ),
        "source": "ASHRAE Handbook — Fundamentals 2017, ch. 4",
        "url":    "https://www.ashrae.org/technical-resources/ashrae-handbook/description-2017-ashrae-handbook-fundamentals",
        "tags":   ["robin", "forced_convection", "air", "h_coefficient"],
        "bc_patterns": ["robin", "dirichlet+robin", "neumann+robin", "dirichlet+neumann+robin"],
    },
    {
        "ref_id":  "bc_robin_h_water",
        "type":    "bc_practice",
        "subject": "water cooling h coefficient",
        "text": (
            "Liquid water cooling: h = 500–10,000 W/(m²·K). "
            "Pool / slow-flow cooling: h ≈ 500–2,000. "
            "Forced flow in channels (Re > 10,000): h ≈ 2,000–8,000. "
            "Boiling heat transfer peak: h ≈ 10,000–100,000 W/(m²·K). "
            "Simulations of water-cooled components with h < 200 are under-estimating "
            "the cooling effect."
        ),
        "source": "Incropera et al., Fundamentals of Heat and Mass Transfer, Table 1.1",
        "url":    "https://www.wiley.com/en-us/Fundamentals+of+Heat+and+Mass+Transfer%2C+7th+Edition-p-9780470501979",
        "tags":   ["robin", "water_cooling", "h_coefficient", "convection"],
        "bc_patterns": ["robin", "dirichlet+robin", "neumann+robin", "dirichlet+neumann+robin"],
    },
    {
        "ref_id":  "bc_neumann_heat_flux_typical",
        "type":    "bc_practice",
        "subject": "typical heat flux magnitudes",
        "text": (
            "Common Neumann BC heat flux values: "
            "Electronic chip (high power): 50–500 W/cm² = 500,000–5,000,000 W/m². "
            "Solar irradiance (direct normal): ~1,000 W/m². "
            "Industrial furnace wall: 10,000–100,000 W/m². "
            "Human body metabolic heat: ~50–100 W/m². "
            "A Neumann value > 1,000,000 W/m² without justification is unusual. "
            "Note: positive q = flux into domain (heat addition)."
        ),
        "source": "CRC Handbook of Chemistry and Physics, 97th ed.; Incropera et al.",
        "url":    "https://www.taylorfrancis.com/books/edit/10.1201/9781315369587/crc-handbook-chemistry-physics",
        "tags":   ["neumann", "heat_flux", "magnitude_check", "boundary_condition"],
        "bc_patterns": ["neumann", "dirichlet+neumann", "neumann+robin", "dirichlet+neumann+robin"],
    },
    {
        "ref_id":  "bc_dirichlet_realistic_temps",
        "type":    "bc_practice",
        "subject": "physically realistic Dirichlet temperature ranges",
        "text": (
            "Dirichlet boundary temperatures should be physically realistic: "
            "Cryogenic: 4–100 K (liquid He/N₂ cooling). "
            "Ambient: 250–320 K. "
            "Industrial process heating: 400–1,500 K. "
            "Combustion walls: 800–2,000 K. "
            "If T_max > 3,000 K in a simulation, verify units (K vs °C) and "
            "that material properties remain valid at those temperatures."
        ),
        "source": "ASME PTC Performance Test Codes / Engineering practice",
        "url":    "https://www.asme.org/codes-standards/find-codes-standards/ptc-performance-test-codes",
        "tags":   ["dirichlet", "temperature_range", "physical_validity"],
        "bc_patterns": ["dirichlet", "dirichlet+neumann", "dirichlet+robin", "dirichlet+neumann+robin"],
    },
]


# ─── Solver guidance references ───────────────────────────────────────────────

SOLVER_REFERENCES: list[dict] = [
    {
        "ref_id":  "solver_mesh_resolution",
        "type":    "solver_guidance",
        "subject": "minimum mesh resolution for accuracy",
        "text": (
            "Rule of thumb: at least 10 elements across the thinnest dimension "
            "of the domain for P1 elements. For P2 elements, 5–6 elements suffice. "
            "For boundary layer effects near Robin BCs, use at least 5 elements "
            "within the thermal boundary layer δ_T = δ/Pr^(1/3). "
            "In this solver: nx=10 on a 1m domain → 0.1m elements, which is "
            "marginal for steep gradients. Use nx ≥ 20 for reliable results."
        ),
        "source": "Brenner & Scott, Mathematical Theory of Finite Element Methods, ch. 5",
        "url":    "https://link.springer.com/book/10.1007/978-0-387-75934-0",
        "tags":   ["mesh", "resolution", "accuracy", "P1", "P2"],
        "bc_patterns": [],
    },
    {
        "ref_id":  "solver_dt_stability",
        "type":    "solver_guidance",
        "subject": "time-step stability criterion (Fourier number)",
        "text": (
            "For explicit time integration the Fourier stability limit is "
            "Δt ≤ Δx²/(2α) in 1D, Δx²/(4α) in 2D. "
            "The backward Euler scheme (θ=1) in this solver is unconditionally stable, "
            "but accuracy still requires Fo = α·Δt/Δx² < 1 for temporal accuracy. "
            "Example: steel (α=1.3×10⁻⁵ m²/s), Δx=0.01m → "
            "Δt_accurate < 0.01²/(1.3×10⁻⁵) ≈ 7.7s. "
            "Using Δt=100s with this mesh loses temporal accuracy."
        ),
        "source": "Ferziger, Perić & Street, Computational Methods for Fluid Dynamics, ch. 7",
        "url":    "https://link.springer.com/book/10.1007/978-3-319-99693-6",
        "tags":   ["timestep", "stability", "Fourier_number", "backward_euler", "accuracy"],
        "bc_patterns": [],
    },
    {
        "ref_id":  "solver_p2_advantage",
        "type":    "solver_guidance",
        "subject": "P2 vs P1 elements: when to upgrade",
        "text": (
            "P2 (quadratic) elements converge as O(h³) in L2 vs O(h²) for P1. "
            "Use P2 when: (a) steep gradients near boundaries (Robin/Dirichlet), "
            "(b) coarse mesh is forced by memory constraints, "
            "(c) post-processing needs accurate heat-flux ∇u (flux is O(h) for P1, "
            "O(h²) for P2). P2 roughly doubles memory usage and assembly time. "
            "For smooth solutions on fine meshes, P1 is usually sufficient."
        ),
        "source": "Logg, Mardal & Wells, Automated Solution of Differential Equations by FEM",
        "url":    "https://link.springer.com/book/10.1007/978-3-642-23099-8",
        "tags":   ["P2", "P1", "element_degree", "convergence", "accuracy"],
        "bc_patterns": [],
    },
    {
        "ref_id":  "solver_cg_vs_gmres",
        "type":    "solver_guidance",
        "subject": "linear solver choice: CG vs GMRES",
        "text": (
            "The heat equation produces a symmetric positive-definite (SPD) system. "
            "Use CG + hypre (AMG) for SPD problems — optimal for elliptic PDEs. "
            "GMRES is needed only for non-symmetric systems (e.g. convection-dominated "
            "or coupled problems). Switching from CG to GMRES for the heat equation "
            "gives no accuracy benefit and ~20–40% overhead from non-symmetric storage. "
            "Keep petsc_solver=cg, petsc_preconditioner=hypre unless the problem changes."
        ),
        "source": "Saad, Iterative Methods for Sparse Linear Systems, 2nd ed.",
        "url":    "https://epubs.siam.org/doi/book/10.1137/1.9780898718003",
        "tags":   ["CG", "GMRES", "linear_solver", "SPD", "preconditioner"],
        "bc_patterns": [],
    },
]


# ─── Domain physics references ─────────────────────────────────────────────────

DOMAIN_REFERENCES: list[dict] = [
    {
        "ref_id":  "domain_micro_radiation",
        "type":    "domain_physics",
        "subject": "radiation effects at micro-scale",
        "text": (
            "At micro-scale (< 1.5 cm), radiation heat transfer is negligible "
            "(Q_rad ∝ L² while Q_cond ∝ ΔT/L). Conduction dominates. "
            "Exception: high-temperature MEMS or laser-heated micro-domains "
            "where T > 1000K — radiation then becomes significant even at μm scale."
        ),
        "source": "Incropera et al., ch. 12 / Majumdar, ASME J. Heat Transfer 115(1):7, 1993",
        "url":    "https://asmedigitalcollection.asme.org/heattransfer/article-abstract/115/1/7/441234/",
        "tags":   ["micro", "radiation", "scale_effects", "validity"],
        "domain_labels": ["micro"],
    },
    {
        "ref_id":  "domain_structural_gravity",
        "type":    "domain_physics",
        "subject": "structural-scale domains and buoyancy",
        "text": (
            "For structural-scale domains (> 20 cm), natural convection in enclosed "
            "air gaps can contribute significantly (Ra ∝ L³). "
            "In a 1m vertical air gap with ΔT=50K: Ra ≈ 5×10⁸ (turbulent). "
            "A pure-conduction model underestimates heat transfer by 5–20× "
            "for large-scale air-coupled domains. Add equivalent convective "
            "k_eff = k × Nu or model flow explicitly."
        ),
        "source": "Churchill & Chu, Int. J. Heat Mass Transfer 18:1323, 1975",
        "url":    "https://www.sciencedirect.com/science/article/abs/pii/0017931075901715",
        "tags":   ["structural", "natural_convection", "buoyancy", "scale_effects"],
        "domain_labels": ["structural"],
    },
    {
        "ref_id":  "domain_panel_thermal_mass",
        "type":    "domain_physics",
        "subject": "thermal time constant for panel-scale components",
        "text": (
            "For a panel-scale component (L ~ 0.1–0.2m), the thermal time constant "
            "τ = ρ·cp·L²/(π²·k). For a steel panel (k=50, ρ=7800, cp=500), L=0.1m: "
            "τ ≈ 7800×500×0.01/(9.87×50) ≈ 79s. "
            "Simulations with t_end << τ show transient effects; "
            "t_end >> τ reaches steady state. "
            "Set t_end ≥ 3τ to reach 95% of steady state."
        ),
        "source": "Carslaw & Jaeger, Conduction of Heat in Solids, 2nd ed., ch. 3",
        "url":    "https://global.oup.com/academic/product/conduction-of-heat-in-solids-9780198533689",
        "tags":   ["panel", "thermal_time_constant", "steady_state", "transient"],
        "domain_labels": ["panel", "component"],
    },
]


# ─── All references combined ──────────────────────────────────────────────────

ALL_REFERENCES: list[dict] = (
    MATERIAL_REFERENCES + BC_REFERENCES + SOLVER_REFERENCES + DOMAIN_REFERENCES
)
