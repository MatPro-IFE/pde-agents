"""
Heat Equation Solver using FEniCSx (DOLFINx)

Solves the transient heat conduction PDE:

    ρ c_p ∂u/∂t - ∇·(k ∇u) = f     in Ω × (0, T]

with boundary conditions:
    u = g                    on Γ_D  (Dirichlet)
    k ∂u/∂n = h              on Γ_N  (Neumann flux)
    k ∂u/∂n = α(u_∞ - u)    on Γ_R  (Robin / convective)

Time integration: Backward Euler (θ-scheme with θ=1)
Spatial discretization: Continuous Galerkin FEM (P1 or P2 elements)

Mesh sources
────────────
Built-in (default):
  Structured rectangular / box meshes via DOLFINx built-in generators.
  BCs reference boundaries by location string: "left", "right", "top",
  "bottom", "front", "back".

Gmsh (optional):
  Complex geometries via the Gmsh Python API.
  Set HeatConfig.geometry = {"type": "<name>", ...} to enable.
  BCs reference boundaries by the Gmsh physical-group name defined in
  simulations/geometry/gmsh_geometries.py.

  Available geometry types:
    rectangle, l_shape, circle, annulus, hollow_rectangle,
    t_shape, stepped_notch   (2D)
    box, cylinder            (3D)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

try:
    import dolfinx
    from dolfinx import mesh as dmesh, fem, io
    from dolfinx.fem import (
        Constant, Function, functionspace, dirichletbc,
        locate_dofs_topological, form,
    )
    from dolfinx.fem.petsc import (
        assemble_matrix, assemble_vector, apply_lifting, set_bc,
    )
    from dolfinx.mesh import (
        create_unit_square, create_unit_cube,
        create_rectangle, create_box,
        locate_entities_boundary,
    )
    import numpy as _np_mesh
    from dolfinx.io import XDMFFile, VTXWriter
    import ufl
    from ufl import TestFunction, TrialFunction, dx, ds, grad, inner
    from mpi4py import MPI
    from petsc4py import PETSc
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False
    print("WARNING: dolfinx not available. Using mock solver for testing.")


# ─── Configuration Dataclass ──────────────────────────────────────────────────

@dataclass
class HeatConfig:
    """Complete configuration for a heat equation simulation run."""

    # Domain
    dim: int = 2                    # spatial dimension (2 or 3)
    nx: int = 32                    # mesh nodes in x
    ny: int = 32                    # mesh nodes in y
    nz: int = 16                    # mesh nodes in z (3D only)
    Lx: float = 1.0                 # physical domain length in x [m]
    Ly: float = 1.0                 # physical domain length in y [m]
    Lz: float = 1.0                 # physical domain length in z [m] (3D only)

    # Physical parameters
    rho: float = 1.0                # density [kg/m³]
    cp: float = 1.0                 # specific heat [J/(kg·K)]
    k: float = 1.0                  # thermal conductivity [W/(m·K)]

    # Source term (constant body heat generation)
    source: float = 0.0             # [W/m³]

    # Boundary conditions
    # Each entry must have "type" and "boundary" (or legacy "location") keys.
    #
    # For built-in meshes, "boundary" is a cardinal name:
    #   "left", "right", "top", "bottom", "front", "back", "all"
    #
    # For Gmsh meshes, "boundary" is the physical-group name defined in
    # the geometry builder (e.g. "inner_wall", "hole_wall", "step_face").
    #
    # BC types:
    #   dirichlet : {"type": "dirichlet", "boundary": "left",  "value": 800.0}
    #   neumann   : {"type": "neumann",   "boundary": "top",   "value": 0.0}
    #               (value = heat flux q [W/m²], positive = into domain)
    #   robin     : {"type": "robin",     "boundary": "right",
    #                "h": 50.0,      # convection coefficient [W/(m²·K)]
    #                "T_inf": 300.0} # far-field temperature [K]
    bcs: list = field(default_factory=lambda: [
        {"type": "dirichlet", "boundary": "left",   "value": 0.0},
        {"type": "dirichlet", "boundary": "right",  "value": 1.0},
        {"type": "neumann",   "boundary": "top",    "value": 0.0},
        {"type": "neumann",   "boundary": "bottom", "value": 0.0},
    ])

    # Gmsh geometry specification (None → use built-in DOLFINx mesh generators)
    # Example: {"type": "l_shape", "Lx": 0.1, "Ly": 0.1,
    #           "cut_x": 0.05, "cut_y": 0.05, "mesh_size": 0.004}
    geometry: Optional[dict] = None

    # Initial condition
    u_init: float = 0.0             # uniform initial temperature

    # Time parameters
    t_start: float = 0.0
    t_end: float = 1.0
    dt: float = 0.01
    theta: float = 1.0              # 1=Backward Euler, 0.5=Crank-Nicolson

    # FEM parameters
    element_degree: int = 1         # polynomial degree (1=P1, 2=P2)

    # Output
    output_dir: str = "/workspace/results"
    run_id: str = "run_001"
    save_every: int = 10            # save every N time steps
    save_format: str = "xdmf"       # "xdmf" or "vtx"

    # Solver
    petsc_solver: str = "cg"
    petsc_preconditioner: str = "hypre"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HeatConfig":
        # Only pass keys that are valid dataclass fields so that
        # serialised dicts with extra keys do not cause TypeErrors.
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "HeatConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ─── Boundary Locators ────────────────────────────────────────────────────────

def _make_locator(location: str, dim: int,
                  Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0,
                  tol: float = 1e-10):
    """Return a callable that identifies boundary facets by name.

    Supports arbitrary domain dimensions [0,Lx] × [0,Ly] × [0,Lz].
    """
    if dim == 2:
        locators = {
            "left":   lambda x: np.isclose(x[0], 0.0, atol=tol),
            "right":  lambda x: np.isclose(x[0], Lx,  atol=tol),
            "bottom": lambda x: np.isclose(x[1], 0.0, atol=tol),
            "top":    lambda x: np.isclose(x[1], Ly,  atol=tol),
            "all":    lambda x: np.ones(x.shape[1], dtype=bool),
        }
    else:
        locators = {
            "left":   lambda x: np.isclose(x[0], 0.0, atol=tol),
            "right":  lambda x: np.isclose(x[0], Lx,  atol=tol),
            "bottom": lambda x: np.isclose(x[1], 0.0, atol=tol),
            "top":    lambda x: np.isclose(x[1], Ly,  atol=tol),
            "front":  lambda x: np.isclose(x[2], 0.0, atol=tol),
            "back":   lambda x: np.isclose(x[2], Lz,  atol=tol),
            "all":    lambda x: np.ones(x.shape[1], dtype=bool),
        }
    if location not in locators:
        raise ValueError(f"Unknown location '{location}'. Valid: {list(locators.keys())}")
    return locators[location]


# ─── Simulation Result ────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    run_id: str
    config: HeatConfig
    status: str                 # "success" | "failed" | "converged"
    wall_time: float            # seconds
    n_dofs: int
    n_timesteps: int
    final_time: float
    max_temperature: float
    min_temperature: float
    mean_temperature: float
    output_files: list[str]
    error_message: str = ""
    convergence_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Run {self.run_id}: {self.status} in {self.wall_time:.2f}s | "
            f"DOFs={self.n_dofs} | T∈[{self.min_temperature:.4f}, {self.max_temperature:.4f}]"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


# ─── Main Solver ──────────────────────────────────────────────────────────────

class HeatEquationSolver:
    """
    FEniCSx solver for the transient heat equation.

    Weak formulation (θ-scheme):
      Find u_h ∈ V_h such that for all v_h ∈ V̂_h:

      (ρ c_p / dt) (u_h - u_n, v)_Ω
      + θ   a(u_h,  v)
      + (1-θ) a(u_n, v)
      = θ   L(v) + (1-θ) L(v)

    where a(u,v) = ∫_Ω k ∇u·∇v dx  and  L(v) = ∫_Ω f v dx + ∫_Γ_N h v ds
    """

    def __init__(self, config: HeatConfig):
        if not DOLFINX_AVAILABLE:
            raise RuntimeError("dolfinx is required. Run inside the fenics-runner container.")
        self.cfg  = config
        self.comm = MPI.COMM_WORLD
        # Set by _build_mesh: None = built-in path, dict = Gmsh path
        self._gmsh_boundary_names: Optional[dict] = None
        self._setup_output_dir()

    def _setup_output_dir(self):
        self.out_dir = Path(self.cfg.output_dir) / self.cfg.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _build_mesh(self):
        """Create mesh — either via Gmsh (if geometry spec provided) or DOLFINx built-ins."""
        cfg = self.cfg

        if cfg.geometry is not None:
            self._build_mesh_gmsh()
            return

        # ── Built-in structured mesh ──────────────────────────────────────────
        self._gmsh_boundary_names = None  # signal: use coordinate locators
        if cfg.dim == 2:
            if cfg.Lx == 1.0 and cfg.Ly == 1.0:
                self.msh = create_unit_square(
                    self.comm, cfg.nx, cfg.ny,
                    cell_type=dmesh.CellType.triangle,
                    ghost_mode=dmesh.GhostMode.shared_facet,
                )
            else:
                self.msh = create_rectangle(
                    self.comm,
                    [_np_mesh.array([0.0, 0.0]), _np_mesh.array([cfg.Lx, cfg.Ly])],
                    [cfg.nx, cfg.ny],
                    cell_type=dmesh.CellType.triangle,
                    ghost_mode=dmesh.GhostMode.shared_facet,
                )
        elif cfg.dim == 3:
            if cfg.Lx == 1.0 and cfg.Ly == 1.0 and cfg.Lz == 1.0:
                self.msh = create_unit_cube(
                    self.comm, cfg.nx, cfg.ny, cfg.nz,
                    cell_type=dmesh.CellType.tetrahedron,
                    ghost_mode=dmesh.GhostMode.shared_facet,
                )
            else:
                self.msh = create_box(
                    self.comm,
                    [_np_mesh.array([0.0, 0.0, 0.0]),
                     _np_mesh.array([cfg.Lx, cfg.Ly, cfg.Lz])],
                    [cfg.nx, cfg.ny, cfg.nz],
                    cell_type=dmesh.CellType.tetrahedron,
                    ghost_mode=dmesh.GhostMode.shared_facet,
                )
        else:
            raise ValueError(f"dim must be 2 or 3, got {cfg.dim}")

    def _build_mesh_gmsh(self):
        """Build mesh from a Gmsh geometry specification."""
        try:
            from simulations.geometry.gmsh_geometries import build_gmsh_geometry
        except ImportError:
            from geometry.gmsh_geometries import build_gmsh_geometry

        spec = dict(self.cfg.geometry)  # copy so we don't mutate config
        # Propagate Lx/Ly/Lz from HeatConfig if not already in spec
        for key in ("Lx", "Ly", "Lz"):
            if key not in spec:
                spec[key] = getattr(self.cfg, key)

        gmsh_result = build_gmsh_geometry(spec)
        self.msh = gmsh_result.mesh
        self._gmsh_facet_tags = gmsh_result.facet_tags      # meshtags from Gmsh
        self._gmsh_boundary_names = gmsh_result.boundary_names  # name → int tag

        geo_type = spec.get("type", "gmsh")
        n_cells  = self.msh.topology.index_map(gmsh_result.dim).size_local
        print(f"  Gmsh geometry : {geo_type}")
        print(f"  Cells         : {n_cells:,}")
        print(f"  Boundaries    : {list(gmsh_result.boundary_names.keys())}")

    def _build_function_spaces(self):
        """Create continuous Galerkin function space."""
        self.V = functionspace(self.msh, ("Lagrange", self.cfg.element_degree))
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.u_h = Function(self.V, name="temperature")    # current solution
        self.u_n = Function(self.V, name="temperature_n")  # previous solution

    def _apply_initial_condition(self):
        """Set uniform initial temperature."""
        self.u_n.x.array[:] = self.cfg.u_init
        self.u_h.x.array[:] = self.cfg.u_init

    def _resolve_boundary_name(self, bc_def: dict) -> str:
        """Return the boundary name from either 'boundary' or legacy 'location' key."""
        name = bc_def.get("boundary") or bc_def.get("location")
        if name is None:
            raise ValueError(
                f"BC definition must include 'boundary' key: {bc_def}"
            )
        return name

    def _build_boundary_conditions(self):
        """Parse and apply Dirichlet, Neumann, and Robin BCs.

        Supports two backends:
        - Built-in mesh: uses coordinate-based locators (legacy 'location' key).
        - Gmsh mesh:     uses integer facet tags from Gmsh physical groups.
        """
        if self._gmsh_boundary_names is not None:
            self._build_boundary_conditions_gmsh()
        else:
            self._build_boundary_conditions_builtin()

    def _build_boundary_conditions_builtin(self):
        """BCs for structured built-in meshes using coordinate locators."""
        cfg = self.cfg
        msh = self.msh
        fdim = msh.topology.dim - 1

        self.dirichlet_bcs = []
        self.neumann_terms = []
        self.robin_terms   = []

        facet_indices_list, facet_markers_list = [], []
        marker_counter = 1
        marker_map = {}

        for bc_def in cfg.bcs:
            bc_type  = bc_def["type"]
            boundary = self._resolve_boundary_name(bc_def)
            locator  = _make_locator(boundary, cfg.dim,
                                     Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz)
            facets   = locate_entities_boundary(msh, fdim, locator)

            if bc_type == "dirichlet":
                val  = Constant(msh, PETSc.ScalarType(bc_def["value"]))
                dofs = locate_dofs_topological(self.V, fdim, facets)
                self.dirichlet_bcs.append(dirichletbc(val, dofs, self.V))

            elif bc_type in ("neumann", "robin"):
                tag = marker_counter
                marker_counter += 1
                marker_map[tag] = bc_def
                if len(facets) > 0:
                    facet_indices_list.append(facets)
                    facet_markers_list.append(
                        np.full(len(facets), tag, dtype=np.int32)
                    )

        if facet_indices_list:
            all_facets  = np.concatenate(facet_indices_list)
            all_markers = np.concatenate(facet_markers_list)
            sorted_idx  = np.argsort(all_facets)
            self.facet_tags = dmesh.meshtags(
                msh, fdim,
                all_facets[sorted_idx],
                all_markers[sorted_idx],
            )
            self.ds_tagged = ufl.Measure(
                "ds", domain=msh, subdomain_data=self.facet_tags
            )
            for tag, bc_def in marker_map.items():
                self._register_natural_bc(msh, bc_def, tag)
        else:
            self.ds_tagged = ufl.Measure("ds", domain=msh)

    def _build_boundary_conditions_gmsh(self):
        """BCs for Gmsh meshes using physical-group integer tags."""
        cfg  = self.cfg
        msh  = self.msh
        fdim = msh.topology.dim - 1
        name_to_tag = self._gmsh_boundary_names  # str → int

        self.dirichlet_bcs = []
        self.neumann_terms = []
        self.robin_terms   = []

        # Use the meshtags produced by Gmsh directly as the subdomain measure
        self.facet_tags = self._gmsh_facet_tags
        self.ds_tagged  = ufl.Measure(
            "ds", domain=msh, subdomain_data=self.facet_tags
        )

        for bc_def in cfg.bcs:
            bc_type  = bc_def["type"]
            boundary = self._resolve_boundary_name(bc_def)

            if boundary not in name_to_tag:
                available = list(name_to_tag.keys())
                raise ValueError(
                    f"Boundary '{boundary}' not found in Gmsh geometry. "
                    f"Available: {available}"
                )
            tag    = name_to_tag[boundary]
            facets = self.facet_tags.indices[self.facet_tags.values == tag]

            if bc_type == "dirichlet":
                val  = Constant(msh, PETSc.ScalarType(bc_def["value"]))
                dofs = locate_dofs_topological(self.V, fdim, facets)
                self.dirichlet_bcs.append(dirichletbc(val, dofs, self.V))

            elif bc_type in ("neumann", "robin"):
                self._register_natural_bc(msh, bc_def, tag)

    def _register_natural_bc(self, msh, bc_def: dict, tag: int):
        """Add a Neumann or Robin term for a given facet tag."""
        bc_type = bc_def["type"]
        if bc_type == "neumann":
            # Support both "value" (legacy) and "h" (heat-flux notation)
            flux = bc_def.get("value", bc_def.get("h", 0.0))
            self.neumann_terms.append(
                (Constant(msh, PETSc.ScalarType(flux)), tag)
            )
        elif bc_type == "robin":
            # Support both old-style (alpha/u_inf) and new-style (h/T_inf)
            alpha = bc_def.get("alpha", bc_def.get("h",     1.0))
            u_inf = bc_def.get("u_inf", bc_def.get("T_inf", 0.0))
            self.robin_terms.append((
                Constant(msh, PETSc.ScalarType(alpha)),
                Constant(msh, PETSc.ScalarType(u_inf)),
                tag,
            ))

    def _build_variational_forms(self):
        """Assemble bilinear and linear forms (θ-scheme)."""
        cfg = self.cfg
        msh = self.msh
        u, v = self.u, self.v
        u_n = self.u_n

        dt    = Constant(msh, PETSc.ScalarType(cfg.dt))
        theta = Constant(msh, PETSc.ScalarType(cfg.theta))
        rho_cp = Constant(msh, PETSc.ScalarType(cfg.rho * cfg.cp))
        k     = Constant(msh, PETSc.ScalarType(cfg.k))
        f     = Constant(msh, PETSc.ScalarType(cfg.source))

        # ── Bilinear form  a(u^{n+1}, v) ─────────────────────────────────────
        # (ρc_p/dt)(u,v) + θ k(∇u,∇v)  [+ θ α(u,v) on Robin faces]
        a_ufl = (
            (rho_cp / dt) * inner(u, v) * dx
            + theta * k * inner(grad(u), grad(v)) * dx
        )
        for alpha, _u_inf, tag in self.robin_terms:
            a_ufl = a_ufl + theta * alpha * inner(u, v) * self.ds_tagged(tag)

        # ── Linear form  L(v) ────────────────────────────────────────────────
        # (ρc_p/dt)(u_n,v) - (1-θ) k(∇u_n,∇v) + (f,v)
        # + Neumann: h v ds
        # + Robin:   α u_∞ v ds − (1−θ) α u_n v ds
        L_ufl = (
            (rho_cp / dt) * inner(u_n, v) * dx
            - (1.0 - theta) * k * inner(grad(u_n), grad(v)) * dx
            + inner(f, v) * dx
        )
        for h_val, tag in self.neumann_terms:
            L_ufl = L_ufl + inner(h_val, v) * self.ds_tagged(tag)
        for alpha, u_inf_c, tag in self.robin_terms:
            L_ufl = (
                L_ufl
                + alpha * inner(u_inf_c, v) * self.ds_tagged(tag)
                - (1.0 - theta) * alpha * inner(u_n, v) * self.ds_tagged(tag)
            )

        self.a = form(a_ufl)
        self.L = form(L_ufl)

    def _create_solver(self) -> PETSc.KSP:
        """Create PETSc KSP linear solver."""
        solver = PETSc.KSP().create(self.comm)
        solver.setType(self.cfg.petsc_solver)
        pc = solver.getPC()
        pc.setType(self.cfg.petsc_preconditioner)
        solver.setTolerances(rtol=1e-8, atol=1e-10, max_it=300)
        return solver

    def solve(self) -> SimulationResult:
        """
        Run the full transient simulation.

        Returns a SimulationResult with statistics and output file paths.
        """
        t0 = time.perf_counter()
        cfg = self.cfg

        print(f"\n{'═'*60}")
        print(f"  Heat Equation Solver  |  Run: {cfg.run_id}")
        print(f"  Dim={cfg.dim}D  |  k={cfg.k}  |  ρcₚ={cfg.rho*cfg.cp}")
        if cfg.geometry:
            print(f"  Mesh: Gmsh / {cfg.geometry.get('type', '?')}")
        else:
            print(f"  Mesh: {cfg.nx}×{cfg.ny}" + (f"×{cfg.nz}" if cfg.dim == 3 else ""))
        print(f"  Time: [0, {cfg.t_end}]  dt={cfg.dt}  θ={cfg.theta}")
        print(f"{'═'*60}")

        self._build_mesh()
        self._build_function_spaces()
        self._apply_initial_condition()
        self._build_boundary_conditions()
        self._build_variational_forms()

        n_dofs = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        print(f"  DOFs: {n_dofs:,}")

        # Assemble system matrix (time-independent for constant k and dt)
        A = assemble_matrix(self.a, bcs=self.dirichlet_bcs)
        A.assemble()

        solver = self._create_solver()
        solver.setOperators(A)

        # Save DOF coordinates once — used by the dashboard for proper interpolation
        coords = self.V.tabulate_dof_coordinates()
        np.save(str(self.out_dir / "dof_coords.npy"), coords)

        output_files = []
        if cfg.save_format == "xdmf":
            xdmf_path = str(self.out_dir / "temperature.xdmf")
            xdmf_file = XDMFFile(self.comm, xdmf_path, "w")
            xdmf_file.write_mesh(self.msh)
            output_files.append(xdmf_path)
        elif cfg.save_format == "vtx":
            vtx_path = str(self.out_dir / "temperature.bp")
            vtx_file = VTXWriter(self.comm, vtx_path, [self.u_h])
            output_files.append(vtx_path)

        # Snapshot directory for dashboard animation
        snapshots_dir = self.out_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)

        t = cfg.t_start
        n_steps = int((cfg.t_end - cfg.t_start) / cfg.dt)
        convergence_history = []
        snapshot_times: list[float] = []
        snap_idx = 0

        print(f"\n  Time-stepping ({n_steps} steps):")
        for step in range(n_steps):
            t += cfg.dt

            # Reassemble RHS (u_n updates each step)
            b = assemble_vector(self.L)
            apply_lifting(b, [self.a], bcs=[self.dirichlet_bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, self.dirichlet_bcs)

            solver.solve(b, self.u_h.x.petsc_vec)
            self.u_h.x.scatter_forward()

            # Update previous solution
            self.u_n.x.array[:] = self.u_h.x.array

            # Track convergence (L2 norm of solution)
            u_arr = self.u_h.x.array
            l2_norm = float(np.sqrt(np.mean(u_arr**2)))
            convergence_history.append(l2_norm)

            if step % cfg.save_every == 0:
                if cfg.save_format == "xdmf":
                    xdmf_file.write_function(self.u_h, t)
                elif cfg.save_format == "vtx":
                    vtx_file.write(t)
                # Save numpy snapshot for dashboard animation
                np.save(str(snapshots_dir / f"u_{snap_idx:04d}.npy"), u_arr.copy())
                snapshot_times.append(t)
                snap_idx += 1
                print(f"  t={t:.4f}  |  L2={l2_norm:.6f}  |  "
                      f"T∈[{u_arr.min():.4f}, {u_arr.max():.4f}]")

        # Close output files
        if cfg.save_format == "xdmf":
            xdmf_file.close()
        elif cfg.save_format == "vtx":
            vtx_file.close()

        # Save snapshot timestamps
        np.save(str(self.out_dir / "snapshot_times.npy"), np.array(snapshot_times))

        # Final statistics
        u_final = self.u_h.x.array
        wall_time = time.perf_counter() - t0

        # Save config alongside results
        cfg.save_json(str(self.out_dir / "config.json"))

        # Save final field as numpy array
        np_path = str(self.out_dir / "u_final.npy")
        np.save(np_path, u_final)
        output_files.append(np_path)

        result = SimulationResult(
            run_id=cfg.run_id,
            config=cfg,
            status="success",
            wall_time=wall_time,
            n_dofs=n_dofs,
            n_timesteps=n_steps,
            final_time=t,
            max_temperature=float(u_final.max()),
            min_temperature=float(u_final.min()),
            mean_temperature=float(u_final.mean()),
            output_files=output_files,
            convergence_history=convergence_history,
        )

        print(f"\n  {'─'*56}")
        print(f"  {result.summary()}")
        print(f"  Output: {self.out_dir}")
        print(f"  {'─'*56}\n")

        # Persist result metadata
        result_json = str(self.out_dir / "result.json")
        with open(result_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        return result


# ─── Convenience factory functions ────────────────────────────────────────────

def run_2d_heat(
    nx: int = 32,
    ny: int = 32,
    k: float = 1.0,
    t_end: float = 0.5,
    dt: float = 0.01,
    run_id: str = "heat_2d",
    output_dir: str = "/workspace/results",
    **kwargs,
) -> SimulationResult:
    """Solve 2D heat equation on [0,1]² with standard BCs."""
    cfg = HeatConfig(
        dim=2, nx=nx, ny=ny, k=k,
        t_end=t_end, dt=dt,
        run_id=run_id, output_dir=output_dir,
        bcs=[
            {"type": "dirichlet", "value": 0.0,  "location": "left"},
            {"type": "dirichlet", "value": 1.0,  "location": "right"},
            {"type": "neumann",   "value": 0.0,  "location": "top"},
            {"type": "neumann",   "value": 0.0,  "location": "bottom"},
        ],
        **kwargs,
    )
    return HeatEquationSolver(cfg).solve()


def run_3d_heat(
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    k: float = 1.0,
    t_end: float = 0.5,
    dt: float = 0.01,
    run_id: str = "heat_3d",
    output_dir: str = "/workspace/results",
    **kwargs,
) -> SimulationResult:
    """Solve 3D heat equation on [0,1]³."""
    cfg = HeatConfig(
        dim=3, nx=nx, ny=ny, nz=nz, k=k,
        t_end=t_end, dt=dt,
        run_id=run_id, output_dir=output_dir,
        bcs=[
            {"type": "dirichlet", "value": 0.0,  "location": "left"},
            {"type": "dirichlet", "value": 1.0,  "location": "right"},
            {"type": "neumann",   "value": 0.0,  "location": "top"},
            {"type": "neumann",   "value": 0.0,  "location": "bottom"},
            {"type": "neumann",   "value": 0.0,  "location": "front"},
            {"type": "neumann",   "value": 0.0,  "location": "back"},
        ],
        **kwargs,
    )
    return HeatEquationSolver(cfg).solve()


def parametric_sweep(
    parameter: str,
    values: list,
    base_config: Optional[HeatConfig] = None,
    output_dir: str = "/workspace/results",
) -> list[SimulationResult]:
    """
    Run a parametric study by varying one parameter.

    Example:
        results = parametric_sweep("k", [0.1, 0.5, 1.0, 2.0, 5.0])
    """
    if base_config is None:
        base_config = HeatConfig(dim=2, nx=24, ny=24, t_end=0.2, dt=0.01)

    results = []
    for i, val in enumerate(values):
        cfg_dict = base_config.to_dict()
        cfg_dict[parameter] = val
        cfg_dict["run_id"] = f"sweep_{parameter}_{val:.4g}".replace(".", "p")
        cfg_dict["output_dir"] = output_dir
        cfg = HeatConfig.from_dict(cfg_dict)
        result = HeatEquationSolver(cfg).solve()
        results.append(result)
        print(f"[{i+1}/{len(values)}] {parameter}={val}: {result.summary()}")

    return results


# ─── Gmsh geometry factory functions ─────────────────────────────────────────

def run_l_shape(
    k: float = 1.0,
    Lx: float = 0.1,
    Ly: float = 0.1,
    cut_x: float = 0.05,
    cut_y: float = 0.05,
    mesh_size: float = 0.004,
    t_end: float = 0.5,
    dt: float = 0.01,
    run_id: str = "l_shape_heat",
    output_dir: str = "/workspace/results",
    **kwargs,
) -> SimulationResult:
    """Heat equation on an L-shaped domain.

    Default BCs: hot left wall (Dirichlet 800 K), convective cooling on all
    other exposed surfaces (Robin h=25, T_inf=300 K).
    """
    bcs = kwargs.pop("bcs", [
        {"type": "dirichlet", "boundary": "left",         "value": 800.0},
        {"type": "robin",     "boundary": "top",          "h": 25.0, "T_inf": 300.0},
        {"type": "robin",     "boundary": "bottom_left",  "h": 25.0, "T_inf": 300.0},
        {"type": "robin",     "boundary": "bottom_right", "h": 25.0, "T_inf": 300.0},
        {"type": "robin",     "boundary": "right",        "h": 25.0, "T_inf": 300.0},
        {"type": "robin",     "boundary": "inner_h",      "h": 25.0, "T_inf": 300.0},
        {"type": "robin",     "boundary": "inner_v",      "h": 25.0, "T_inf": 300.0},
    ])
    cfg = HeatConfig(
        dim=2, k=k, Lx=Lx, Ly=Ly, t_end=t_end, dt=dt,
        run_id=run_id, output_dir=output_dir,
        geometry={"type": "l_shape", "Lx": Lx, "Ly": Ly,
                  "cut_x": cut_x, "cut_y": cut_y, "mesh_size": mesh_size},
        bcs=bcs,
        **kwargs,
    )
    return HeatEquationSolver(cfg).solve()


def run_annulus(
    k: float = 50.0,
    r_in: float = 0.01,
    r_out: float = 0.05,
    mesh_size: float = 0.003,
    t_end: float = 1.0,
    dt: float = 0.02,
    run_id: str = "annulus_heat",
    output_dir: str = "/workspace/results",
    **kwargs,
) -> SimulationResult:
    """Radial heat conduction in a pipe cross-section (annular ring).

    Default BCs: fixed inner wall temperature (hot), convective outer wall.
    """
    bcs = kwargs.pop("bcs", [
        {"type": "dirichlet", "boundary": "inner_wall", "value": 600.0},
        {"type": "robin",     "boundary": "outer_wall", "h": 100.0, "T_inf": 300.0},
    ])
    cfg = HeatConfig(
        dim=2, k=k, t_end=t_end, dt=dt,
        run_id=run_id, output_dir=output_dir,
        geometry={"type": "annulus", "r_in": r_in, "r_out": r_out,
                  "mesh_size": mesh_size},
        bcs=bcs,
        **kwargs,
    )
    return HeatEquationSolver(cfg).solve()


def run_hollow_rectangle(
    k: float = 45.0,
    Lx: float = 0.1,
    Ly: float = 0.08,
    hole_w: float = 0.03,
    hole_h: float = 0.025,
    mesh_size: float = 0.004,
    t_end: float = 0.5,
    dt: float = 0.01,
    run_id: str = "hollow_rect_heat",
    output_dir: str = "/workspace/results",
    **kwargs,
) -> SimulationResult:
    """Heat equation on a rectangle with a central rectangular void.

    Default BCs: hot left wall, cold right wall, insulated top/bottom, hot hole wall.
    """
    bcs = kwargs.pop("bcs", [
        {"type": "dirichlet", "boundary": "left",      "value": 800.0},
        {"type": "dirichlet", "boundary": "right",     "value": 300.0},
        {"type": "neumann",   "boundary": "top",       "value": 0.0},
        {"type": "neumann",   "boundary": "bottom",    "value": 0.0},
        {"type": "dirichlet", "boundary": "hole_wall", "value": 600.0},
    ])
    cfg = HeatConfig(
        dim=2, k=k, Lx=Lx, Ly=Ly, t_end=t_end, dt=dt,
        run_id=run_id, output_dir=output_dir,
        geometry={"type": "hollow_rectangle", "Lx": Lx, "Ly": Ly,
                  "hole_w": hole_w, "hole_h": hole_h,
                  "mesh_size": mesh_size},
        bcs=bcs,
        **kwargs,
    )
    return HeatEquationSolver(cfg).solve()


def run_cylinder_3d(
    k: float = 50.0,
    radius: float = 0.05,
    height: float = 0.1,
    mesh_size: float = 0.01,
    t_end: float = 1.0,
    dt: float = 0.02,
    run_id: str = "cylinder_3d_heat",
    output_dir: str = "/workspace/results",
    **kwargs,
) -> SimulationResult:
    """3D heat conduction in a solid cylinder.

    Default BCs: fixed base temperature, convective lateral and top surfaces.
    """
    bcs = kwargs.pop("bcs", [
        {"type": "dirichlet", "boundary": "bottom_face",  "value": 800.0},
        {"type": "robin",     "boundary": "lateral_wall", "h": 50.0, "T_inf": 300.0},
        {"type": "robin",     "boundary": "top_face",     "h": 50.0, "T_inf": 300.0},
    ])
    cfg = HeatConfig(
        dim=3, k=k, t_end=t_end, dt=dt,
        run_id=run_id, output_dir=output_dir,
        geometry={"type": "cylinder", "radius": radius,
                  "height": height, "mesh_size": mesh_size},
        bcs=bcs,
        **kwargs,
    )
    return HeatEquationSolver(cfg).solve()


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="FEniCSx Heat Equation Solver")
    parser.add_argument("--dim",      type=int,   default=2)
    parser.add_argument("--nx",       type=int,   default=32)
    parser.add_argument("--ny",       type=int,   default=32)
    parser.add_argument("--nz",       type=int,   default=16)
    parser.add_argument("--k",        type=float, default=1.0)
    parser.add_argument("--rho",      type=float, default=1.0)
    parser.add_argument("--cp",       type=float, default=1.0)
    parser.add_argument("--source",   type=float, default=0.0)
    parser.add_argument("--t-end",    type=float, default=0.5)
    parser.add_argument("--dt",       type=float, default=0.01)
    parser.add_argument("--theta",    type=float, default=1.0)
    parser.add_argument("--degree",   type=int,   default=1)
    parser.add_argument("--run-id",   type=str,   default="run_001")
    parser.add_argument("--out-dir",  type=str,   default="/workspace/results")
    parser.add_argument("--config",   type=str,   default=None,
                        help="Path to JSON config file (overrides all flags)")
    args = parser.parse_args()

    if args.config:
        cfg = HeatConfig.from_json(args.config)
    else:
        cfg = HeatConfig(
            dim=args.dim, nx=args.nx, ny=args.ny, nz=args.nz,
            k=args.k, rho=args.rho, cp=args.cp, source=args.source,
            t_end=args.t_end, dt=args.dt, theta=args.theta,
            element_degree=args.degree,
            run_id=args.run_id, output_dir=args.out_dir,
        )

    result = HeatEquationSolver(cfg).solve()
    sys.exit(0 if result.status == "success" else 1)
