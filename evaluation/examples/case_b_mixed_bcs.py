"""Case B — Steady heat conduction with mixed boundary conditions.

  Domain : [0,1]²  (uniform triangular mesh, N=64)
  PDE    : -∇·(k ∇T) = Q,   k = 50 W/(m·K) (AISI 1010 steel), Q = 5000 W/m³
  BCs    : T = 300 K  on x=0 (left, Dirichlet)
           -k ∂T/∂n = q_N = 2000 W/m²  on y=1 (top, inward Neumann flux)
           -k ∂T/∂n = h(T-T_inf)  on x=1 (right, Robin/convection),
                       h = 25 W/(m²·K), T_inf = 293 K
           ∂T/∂n = 0  on y=0 (bottom, insulated — natural BC)

This case demonstrates the agent's ability to handle multiple BC types
on a single domain, a common requirement in engineering practice.

Outputs
-------
  output/case_b.npz   — mesh coordinates, triangulation, solution values
  output/case_b.png   — standalone colour-map plot
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
k_steel = 50.0
Q_vol = 5000.0
q_neumann = 2000.0
h_conv = 25.0
T_inf = 293.0

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = k_steel * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(Q_vol))
L = f * v * ufl.dx

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

def top_boundary(x):
    return np.isclose(x[1], 1.0)

def right_boundary(x):
    return np.isclose(x[0], 1.0)

def left_boundary(x):
    return np.isclose(x[0], 0.0)

facets_top = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
facets_right = dolfinx.mesh.locate_entities_boundary(mesh, fdim, right_boundary)

tag_top, tag_right = 1, 2
marked_facets = np.hstack([facets_top, facets_right])
marked_values = np.hstack([
    np.full_like(facets_top, tag_top, dtype=np.int32),
    np.full_like(facets_right, tag_right, dtype=np.int32),
])
sort_idx = np.argsort(marked_facets)
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, marked_facets[sort_idx], marked_values[sort_idx]
)

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

L += PETSc.ScalarType(q_neumann) * v * ds(tag_top)

a += PETSc.ScalarType(h_conv) * T * v * ds(tag_right)
L += PETSc.ScalarType(h_conv * T_inf) * v * ds(tag_right)

bc_left = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(300.0),
    dolfinx.fem.locate_dofs_geometrical(V, left_boundary),
    V,
)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, petsc_options_prefix="case_b",
    bcs=[bc_left],
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
    OUT / "case_b.npz",
    coords=coords, cells=tri_cells, values=vals,
    title="Mixed BCs (steel plate)",
    description="Dirichlet left (300K), Neumann top (2kW/m²), "
                "Robin right (h=25, T_inf=293K), insulated bottom, "
                "Q=5kW/m³, k=50 W/(m·K), N=64",
)

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tri_cells)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
tcf = ax.tricontourf(triang, vals, levels=32, cmap="inferno")
fig.colorbar(tcf, ax=ax, label="Temperature (K)", shrink=0.85)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("(b) Mixed BCs — steel plate", fontsize=10)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "case_b.png", dpi=200)
plt.close(fig)
print(f"Case B done — T range [{vals.min():.2f}, {vals.max():.2f}]")
