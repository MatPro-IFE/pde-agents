"""Case C — Transient heat conduction on an L-shaped domain.

  Domain : L-shape = [0,1]² \\ [0.5,1]²  (Gmsh, mesh size ~0.02)
  PDE    : ρ c_p ∂T/∂t - ∇·(k ∇T) = 0
           ρ = 2700 kg/m³, c_p = 900 J/(kg·K), k = 205 W/(m·K)  (aluminium)
  IC     : T(x,0) = 293 K  everywhere
  BCs    : T = 500 K  on the left edge (x=0)
           T = 293 K  on the bottom edge (y=0)
           ∂T/∂n = 0  on all other boundaries (insulated)
  Time   : t_end = 0.5 s, dt = 0.005 s  (implicit Euler, 100 steps)

The L-shape geometry demonstrates Gmsh meshing of non-rectangular domains.
We save snapshots at t = {0.05, 0.15, 0.50} s.

Outputs
-------
  output/case_c.npz      — mesh coords, cells, solution at 3 time snapshots
  output/case_c.png      — standalone 3-panel plot
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")

import dolfinx
import dolfinx.fem.petsc
import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

rho = 2700.0
cp = 900.0
k_al = 205.0
T_hot = 500.0
T_cold = 293.0
dt = 0.005
t_end = 0.50
snapshot_times = [0.05, 0.15, 0.50]
lc = 0.02

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("lshape")
p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc)
p2 = gmsh.model.occ.addPoint(1.0, 0.0, 0.0, lc)
p3 = gmsh.model.occ.addPoint(1.0, 0.5, 0.0, lc)
p4 = gmsh.model.occ.addPoint(0.5, 0.5, 0.0, lc)
p5 = gmsh.model.occ.addPoint(0.5, 1.0, 0.0, lc)
p6 = gmsh.model.occ.addPoint(0.0, 1.0, 0.0, lc)

lines = [
    gmsh.model.occ.addLine(p1, p2),
    gmsh.model.occ.addLine(p2, p3),
    gmsh.model.occ.addLine(p3, p4),
    gmsh.model.occ.addLine(p4, p5),
    gmsh.model.occ.addLine(p5, p6),
    gmsh.model.occ.addLine(p6, p1),
]
cl = gmsh.model.occ.addCurveLoop(lines)
surf = gmsh.model.occ.addPlaneSurface([cl])
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [surf], tag=1)
gmsh.model.setPhysicalName(2, 1, "domain")
gmsh.model.mesh.generate(2)

mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
mesh = mesh_data.mesh
gmsh.finalize()

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

T_n = dolfinx.fem.Function(V, name="T_n")
T_n.x.array[:] = T_cold

T_h = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

alpha = k_al / (rho * cp)
dt_c = dolfinx.fem.Constant(mesh, PETSc.ScalarType(dt))

a = T_h * v * ufl.dx + dt_c * alpha * ufl.inner(ufl.grad(T_h), ufl.grad(v)) * ufl.dx
L_form = T_n * v * ufl.dx

def left_edge(x):
    return np.isclose(x[0], 0.0)

def bottom_edge(x):
    return np.isclose(x[1], 0.0)

bc_hot = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_hot),
    dolfinx.fem.locate_dofs_geometrical(V, left_edge),
    V,
)
bc_cold = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_cold),
    dolfinx.fem.locate_dofs_geometrical(V, bottom_edge),
    V,
)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L_form, petsc_options_prefix="case_c",
    bcs=[bc_hot, bc_cold],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

coords = mesh.geometry.x[:, :2]
cells_raw = mesh.geometry.dofmap
tri_cells = np.array(
    [[cells_raw[i][0], cells_raw[i][1], cells_raw[i][2]]
     for i in range(len(cells_raw))]
)

snapshots = {}
t = 0.0
step = 0
snap_idx = 0
while t < t_end - 1e-12:
    t += dt
    step += 1
    Th = problem.solve()
    T_n.x.array[:] = Th.x.array
    if snap_idx < len(snapshot_times) and t >= snapshot_times[snap_idx] - 1e-12:
        snapshots[f"t_{snapshot_times[snap_idx]:.2f}"] = Th.x.array.real.copy()
        print(f"  snapshot t={t:.3f}s — T [{Th.x.array.real.min():.1f}, {Th.x.array.real.max():.1f}]")
        snap_idx += 1

np.savez(
    OUT / "case_c.npz",
    coords=coords, cells=tri_cells,
    **snapshots,
    snapshot_times=np.array(snapshot_times),
    title="Transient L-shape (aluminium)",
    description=f"k={k_al}, rho={rho}, cp={cp}, dt={dt}, lc={lc}, "
                f"T_hot={T_hot}, T_cold={T_cold}",
)

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tri_cells)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for i, tsnap in enumerate(snapshot_times):
    key = f"t_{tsnap:.2f}"
    ax = axes[i]
    tcf = ax.tricontourf(triang, snapshots[key], levels=32, cmap="inferno")
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    ax.set_title(f"t = {tsnap:.2f} s", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    if i == 0:
        ax.set_ylabel("y (m)")
fig.suptitle("(c) Transient heat — L-shaped aluminium domain", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "case_c.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Case C done — {step} time steps")
