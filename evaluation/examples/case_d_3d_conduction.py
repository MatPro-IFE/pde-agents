"""Case D — 3D steady-state heat conduction on a unit cube.

  Domain : [0,1]³  (tetrahedral mesh, N=24 per edge)
  PDE    : -∇·(k ∇T) = Q,  k = 16.3 W/(m·K) (stainless steel 304), Q = 5000 W/m³
  BCs    : T = 300 K  on z=0 (bottom face, Dirichlet)
           T = 600 K  on x=0 (left face, Dirichlet — e.g. furnace wall)
           -k ∂T/∂n = h(T-T_inf)  on z=1 (top face, Robin/convection),
                       h = 50 W/(m²·K), T_inf = 293 K
           ∂T/∂n = 0  on remaining three lateral faces (insulated)

The asymmetric BCs produce a genuinely 3D temperature field with
gradients in both x and z, demonstrating the system's 3D FEM capability.
We render three slices (z=0.0, z=0.5, z=1.0).

Outputs
-------
  output/case_d.npz   — vertex coordinates, tetrahedra, solution values, slices
  output/case_d.png   — 3-panel cross-section plot (z = 0.0, 0.5, 1.0)
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")

import dolfinx
import dolfinx.fem.petsc
import dolfinx.geometry
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

N = 24
k_ss = 16.3
Q_vol = 5_000.0
h_conv = 50.0
T_inf = 293.0
T_base = 300.0
T_hot_face = 600.0

mesh = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, N, N, N, dolfinx.mesh.CellType.tetrahedron
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = k_ss * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(Q_vol))
L = f * v * ufl.dx

fdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(fdim, mesh.topology.dim)

def bottom_face(x):
    return np.isclose(x[2], 0.0)

def top_face(x):
    return np.isclose(x[2], 1.0)

def left_face(x):
    return np.isclose(x[0], 0.0)

facets_top = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_face)
tag_top = 1
facet_tags = dolfinx.mesh.meshtags(
    mesh, fdim, facets_top.astype(np.int32),
    np.full_like(facets_top, tag_top, dtype=np.int32),
)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

a += PETSc.ScalarType(h_conv) * T * v * ds(tag_top)
L += PETSc.ScalarType(h_conv * T_inf) * v * ds(tag_top)

bc_bottom = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_base),
    dolfinx.fem.locate_dofs_geometrical(V, bottom_face),
    V,
)
bc_left = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_hot_face),
    dolfinx.fem.locate_dofs_geometrical(V, left_face),
    V,
)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, petsc_options_prefix="case_d",
    bcs=[bc_bottom, bc_left],
    petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                   "ksp_rtol": "1e-10"},
)
Th = problem.solve()

coords = mesh.geometry.x  # (n_verts, 3)
vals = Th.x.array.real

grid_n = 100
xi = np.linspace(0, 1, grid_n)
yi = np.linspace(0, 1, grid_n)
Xi, Yi = np.meshgrid(xi, yi)

slice_z = [0.0, 0.5, 1.0]
slice_data = {}

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
for zval in slice_z:
    Zi = np.full_like(Xi, np.nan)
    pts_3d = np.column_stack([Xi.ravel(), Yi.ravel(),
                              np.full(grid_n * grid_n, zval)])
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pts_3d)
    cell_ids = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, pts_3d
    )
    for j in range(len(pts_3d)):
        cells_j = cell_ids.links(j)
        if len(cells_j) > 0:
            val = Th.eval(pts_3d[j], cells_j[0])
            row, col = divmod(j, grid_n)
            Zi[row, col] = val[0]
    slice_data[f"z_{zval:.1f}"] = Zi

np.savez(
    OUT / "case_d.npz",
    coords=coords, values=vals,
    Xi=Xi, Yi=Yi,
    **{k: v for k, v in slice_data.items()},
    slice_z=np.array(slice_z),
    title="3D steady conduction (stainless steel 304)",
    description=f"k={k_ss}, Q={Q_vol}, h={h_conv}, T_inf={T_inf}, "
                f"T_base={T_base}, N={N}",
)

vmin, vmax = vals.min(), vals.max()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
for i, zval in enumerate(slice_z):
    Zi = slice_data[f"z_{zval:.1f}"]
    ax = axes[i]
    cf = ax.contourf(Xi, Yi, Zi, levels=32, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(f"z = {zval:.1f}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    if i == 0:
        ax.set_ylabel("y (m)")

fig.colorbar(cf, ax=axes.tolist(), label="Temperature (K)", shrink=0.85,
             pad=0.03)
fig.suptitle("(d) 3D steady conduction — stainless steel cube, cross-sections",
             fontsize=11, y=1.02)
fig.savefig(OUT / "case_d.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Case D done — T range [{vals.min():.2f}, {vals.max():.2f}], "
      f"{len(vals)} vertices")
