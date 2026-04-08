"""Case A — Steady heat conduction on a unit square with Dirichlet BCs.

  Domain : [0,1]²  (uniform triangular mesh, N=64)
  PDE    : -∇·(k ∇T) = 0,  k = 385 W/(m·K)  (copper)
  BCs    : T = 373.15 K on x=0 (left),  T = 273.15 K on x=1 (right)
  Exact  : T(x) = 373.15 − 100·x  — linear gradient

Outputs
-------
  output/case_a.npz   — mesh coordinates, triangulation, solution values
  output/case_a.png   — standalone colour-map plot
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

N = 64
k_copper = 385.0

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = k_copper * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
L = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

def left(x):
    return np.isclose(x[0], 0.0)

def right(x):
    return np.isclose(x[0], 1.0)

T_hot = 373.15
T_cold = 273.15

bc_left = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_hot),
    dolfinx.fem.locate_dofs_geometrical(V, left),
    V,
)
bc_right = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_cold),
    dolfinx.fem.locate_dofs_geometrical(V, right),
    V,
)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, petsc_options_prefix="case_a",
    bcs=[bc_left, bc_right],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
Th = problem.solve()

coords = mesh.geometry.x[:, :2]
cells = mesh.geometry.dofmap
tri_cells = np.array(
    [[cells[i][0], cells[i][1], cells[i][2]] for i in range(len(cells))]
)
vals = Th.x.array.real

np.savez(
    OUT / "case_a.npz",
    coords=coords, cells=tri_cells, values=vals,
    title="Steady Dirichlet (copper plate)",
    description=f"T={T_hot}K left, T={T_cold}K right, k=385 W/(m·K), N=64",
)

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tri_cells)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
tcf = ax.tricontourf(triang, vals, levels=32, cmap="inferno")
fig.colorbar(tcf, ax=ax, label="Temperature (K)", shrink=0.85)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("(a) Steady Dirichlet — copper plate", fontsize=10)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "case_a.png", dpi=200)
plt.close(fig)
print(f"Case A done — T range [{vals.min():.2f}, {vals.max():.2f}]")
