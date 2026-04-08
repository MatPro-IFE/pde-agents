"""Case F — Steady heat conduction with a localised Gaussian heat source.

  Domain : [0,1]²  (uniform triangular mesh, N=80)
  PDE    : -∇·(k ∇T) = Q(x,y),  k = 205 W/(m·K)  (aluminium 6061)
           Q(x,y) = Q_peak · exp(-((x-x0)² + (y-y0)²) / (2σ²))
           Q_peak = 5×10⁶ W/m³,  (x0,y0) = (0.3, 0.7),  σ = 0.08
  BCs    : T = 293 K  on all four boundaries (Dirichlet)

The Gaussian heat source produces a localised hot spot that diffuses
outward, demonstrating the system's handling of spatially-varying
volumetric source terms — common in laser heating, Joule heating,
and nuclear applications.

Outputs
-------
  output/case_f.npz   — mesh + solution
  output/case_f.png   — standalone colour-map
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")

import dolfinx
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

N = 80
k_al = 205.0
Q_peak = 5.0e6
x0, y0, sigma = 0.3, 0.7, 0.08
T_boundary = 293.0

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = k_al * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

x = ufl.SpatialCoordinate(mesh)
Q_expr = Q_peak * ufl.exp(
    -((x[0] - x0)**2 + (x[1] - y0)**2) / (2.0 * sigma**2)
)
L = Q_expr * v * ufl.dx

def all_boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
            np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

bc = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_boundary),
    dolfinx.fem.locate_dofs_geometrical(V, all_boundary), V,
)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, petsc_options_prefix="case_f",
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
Th = problem.solve()

coords = mesh.geometry.x[:, :2]
cells_raw = mesh.geometry.dofmap
tri_cells = np.array(
    [[cells_raw[i][0], cells_raw[i][1], cells_raw[i][2]]
     for i in range(len(cells_raw))]
)
vals = Th.x.array.real

np.savez(
    OUT / "case_f.npz",
    coords=coords, cells=tri_cells, values=vals,
    title="Gaussian heat source (Al 6061)",
    description=f"k={k_al}, Q_peak={Q_peak}, centre=({x0},{y0}), "
                f"sigma={sigma}, T_boundary={T_boundary}K, N={N}",
)

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tri_cells)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
tcf = ax.tricontourf(triang, vals, levels=32, cmap="inferno")
fig.colorbar(tcf, ax=ax, label="Temperature (K)", shrink=0.85)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("(f) Gaussian source — Al 6061", fontsize=10)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "case_f.png", dpi=200)
plt.close(fig)
print(f"Case F done — T range [{vals.min():.2f}, {vals.max():.2f}], "
      f"hot-spot peak at ({x0}, {y0})")
