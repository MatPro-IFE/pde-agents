"""Case E — Steady heat conduction on a plate with a circular hole.

  Domain : [0,1]² with a circular cutout of radius 0.15 centred at (0.5, 0.5)
           Meshed with Gmsh (boolean difference), lc ≈ 0.015
  PDE    : -∇·(k ∇T) = 0,  k = 6.7 W/(m·K)  (Ti-6Al-4V titanium alloy)
  BCs    : T = 573 K  on x=0 (left edge)
           T = 293 K  on x=1 (right edge)
           ∂T/∂n = 0  on top, bottom (insulated)
           ∂T/∂n = 0  on the hole boundary (insulated)

The circular hole distorts the otherwise-linear temperature field,
creating thermal concentration zones around the cutout — analogous to
stress concentrations in structural mechanics.  This demonstrates the
system's ability to handle Gmsh boolean geometry operations.

Outputs
-------
  output/case_e.npz   — mesh + solution
  output/case_e.png   — standalone colour-map
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")

import dolfinx
import dolfinx.fem.petsc
import dolfinx.io.gmsh
import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

k_ti = 6.7
T_hot = 573.0
T_cold = 293.0
lc = 0.015
hole_r = 0.15
hole_cx, hole_cy = 0.5, 0.5

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("plate_hole")

rect = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
hole = gmsh.model.occ.addDisk(hole_cx, hole_cy, 0, hole_r, hole_r)
cut_result = gmsh.model.occ.cut([(2, rect)], [(2, hole)])
gmsh.model.occ.synchronize()

surfs = gmsh.model.getEntities(2)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1)
gmsh.model.setPhysicalName(2, 1, "plate")

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)

mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
mesh = mesh_data.mesh
gmsh.finalize()

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = k_ti * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
L = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

def left_edge(x):
    return np.isclose(x[0], 0.0)

def right_edge(x):
    return np.isclose(x[0], 1.0)

bc_hot = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_hot),
    dolfinx.fem.locate_dofs_geometrical(V, left_edge), V,
)
bc_cold = dolfinx.fem.dirichletbc(
    PETSc.ScalarType(T_cold),
    dolfinx.fem.locate_dofs_geometrical(V, right_edge), V,
)

problem = dolfinx.fem.petsc.LinearProblem(
    a, L, petsc_options_prefix="case_e",
    bcs=[bc_hot, bc_cold],
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
    OUT / "case_e.npz",
    coords=coords, cells=tri_cells, values=vals,
    title="Plate with hole (Ti-6Al-4V)",
    description=f"k={k_ti}, T_hot={T_hot}K, T_cold={T_cold}K, "
                f"hole r={hole_r} at ({hole_cx},{hole_cy}), lc={lc}",
)

triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tri_cells)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
tcf = ax.tricontourf(triang, vals, levels=32, cmap="inferno")
fig.colorbar(tcf, ax=ax, label="Temperature (K)", shrink=0.85)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("(e) Plate with hole — Ti-6Al-4V", fontsize=10)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "case_e.png", dpi=200)
plt.close(fig)
print(f"Case E done — T range [{vals.min():.2f}, {vals.max():.2f}], "
      f"{len(vals)} vertices")
