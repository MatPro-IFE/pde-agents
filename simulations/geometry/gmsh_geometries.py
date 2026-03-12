"""
Gmsh geometry builders for FEniCSx / DOLFINx heat equation simulations.

Each builder creates a Gmsh model, tags all boundary segments with named
physical groups, generates the mesh, and returns a GmshMeshResult that the
heat equation solver uses directly.

Available geometry types
────────────────────────
2D geometries (dim=2):
  rectangle        — axis-aligned rectangle [0,Lx]×[0,Ly]
  l_shape          — L-shaped domain (rectangle with corner cut out)
  circle           — circular disk, radius r
  annulus          — ring between inner radius r_in and outer radius r_out
  hollow_rectangle — rectangle with a rectangular hole
  t_shape          — T-shaped cross section (web + flange)
  stepped_notch    — rectangular bar with a step/notch on one side

3D geometries (dim=3):
  box              — axis-aligned box [0,Lx]×[0,Ly]×[0,Lz]
  cylinder         — circular cylinder, radius r, height h

Boundary names
──────────────
Each geometry documents which boundary names are available.
Use these in HeatConfig.bcs[*]["boundary"].

  rectangle:        left, right, bottom, top
  l_shape:          left, bottom_left, bottom_right, right, inner_h, inner_v, top
  circle:           wall
  annulus:          inner_wall, outer_wall
  hollow_rectangle: left, right, bottom, top, hole_wall
  t_shape:          bottom_left, bottom_right, left, right, top,
                    inner_left, inner_right
  stepped_notch:    left, right, bottom, top, step_face
  box:              left, right, front, back, bottom, top
  cylinder:         lateral_wall, bottom_face, top_face
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

try:
    import gmsh
    from dolfinx.io import gmsh as gmsh_io
    from mpi4py import MPI
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class GmshMeshResult:
    """Mesh + boundary/cell tag data produced by a Gmsh geometry builder."""
    mesh: object                    # dolfinx.mesh.Mesh
    facet_tags: object              # dolfinx.mesh.MeshTags  (boundary integer tags)
    cell_tags: object               # dolfinx.mesh.MeshTags  (domain integer tags)
    boundary_names: dict[str, int]  # name → physical group integer tag
    dim: int                        # spatial dimension (2 or 3)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _init_gmsh(model_name: str) -> None:
    """Initialise Gmsh, suppressing verbose mesh-generation output."""
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)   # suppress stdout
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add(model_name)


def _to_mesh(gdim: int) -> GmshMeshResult:
    """Generate the mesh and convert to DOLFINx MeshData."""
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")
    data = gmsh_io.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim)
    gmsh.finalize()

    # Build name → int tag mapping from Gmsh physical groups.
    # data.physical_groups is a dict[str, PhysicalGroup(dim, tag)].
    boundary_names: dict[str, int] = {}
    if data.physical_groups:
        for name, pg in data.physical_groups.items():
            if pg.dim == gdim - 1:           # facets only (curves in 2D, surfaces in 3D)
                boundary_names[name] = pg.tag

    return GmshMeshResult(
        mesh=data.mesh,
        facet_tags=data.facet_tags,
        cell_tags=data.cell_tags,
        boundary_names=boundary_names,
        dim=gdim,
    )


# ─── 2D Geometries ────────────────────────────────────────────────────────────

def _register_boundary_groups(
    curves: list,
    classifier,     # callable(cx, cy) → str
    tag_start: int = 10,
) -> dict[str, int]:
    """
    Group 2D boundary curves by name, then register each group as one Gmsh
    physical group so duplicate names (e.g. all four sides of a hole) are
    merged correctly.

    Returns boundary_names dict: name → Gmsh tag.
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for _, ctag in curves:
        com  = gmsh.model.occ.getCenterOfMass(1, ctag)
        name = classifier(com[0], com[1], ctag)
        groups[name].append(ctag)

    bn: dict[str, int] = {}
    tag = tag_start
    for name, ctags in groups.items():
        gmsh.model.addPhysicalGroup(1, ctags, tag=tag, name=name)
        bn[name] = tag
        tag += 1
    return bn


def _register_surface_groups(
    boundary_surfs: list,
    classifier,      # callable(cx, cy, cz, stag) → str
    tag_start: int = 10,
) -> dict[str, int]:
    """Like _register_boundary_groups but for 3D surface boundaries."""
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for _, stag in boundary_surfs:
        com  = gmsh.model.occ.getCenterOfMass(2, stag)
        name = classifier(com[0], com[1], com[2], stag)
        groups[name].append(stag)

    bn: dict[str, int] = {}
    tag = tag_start
    for name, stags in groups.items():
        gmsh.model.addPhysicalGroup(2, stags, tag=tag, name=name)
        bn[name] = tag
        tag += 1
    return bn


def _build_rectangle(spec: dict) -> GmshMeshResult:
    """
    Axis-aligned rectangle [0, Lx] × [0, Ly].

    Boundaries:  left (x=0), right (x=Lx), bottom (y=0), top (y=Ly)

    Parameters
    ----------
    Lx, Ly      : physical dimensions [m]
    mesh_size   : target element size (default: min(Lx,Ly)/20)
    """
    Lx = float(spec.get("Lx", 1.0))
    Ly = float(spec.get("Ly", 1.0))
    ms = float(spec.get("mesh_size", min(Lx, Ly) / 20.0))

    _init_gmsh("rectangle")
    gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly, tag=1)
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    tol = min(Lx, Ly) * 1e-3
    def classify(cx, cy, _ctag):
        if   abs(cx)      < tol:  return "left"
        elif abs(cx - Lx) < tol:  return "right"
        elif abs(cy)      < tol:  return "bottom"
        elif abs(cy - Ly) < tol:  return "top"
        return "boundary"

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


def _build_l_shape(spec: dict) -> GmshMeshResult:
    """
    L-shaped domain: full rectangle minus the top-right corner rectangle.

         ┌──────┐
         │ cut  │ cut_y
    ┌────┘      │
    │           │
    └───────────┘
    ←─── Lx ───→    ↑ Ly
    cut_x from right, cut_y from top

    Boundaries: left, bottom, right, inner_h (horizontal step),
                inner_v (vertical step), top
    """
    Lx    = float(spec.get("Lx",    0.1))
    Ly    = float(spec.get("Ly",    0.1))
    cut_x = float(spec.get("cut_x", Lx / 2))
    cut_y = float(spec.get("cut_y", Ly / 2))
    ms    = float(spec.get("mesh_size", min(Lx, Ly) / 20.0))

    _init_gmsh("l_shape")
    full = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    cut  = gmsh.model.occ.addRectangle(Lx - cut_x, Ly - cut_y, 0, cut_x, cut_y)
    gmsh.model.occ.cut([(2, full)], [(2, cut)])
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    tol = min(Lx, Ly) * 1e-3
    def classify(cx, cy, _ctag):
        if   abs(cx)              < tol:  return "left"
        elif abs(cy)              < tol:  return "bottom"
        elif abs(cx - Lx)         < tol:  return "right"
        elif abs(cy - (Ly-cut_y)) < tol:  return "inner_h"
        elif abs(cx - (Lx-cut_x)) < tol:  return "inner_v"
        elif abs(cy - Ly)         < tol:  return "top"
        return "boundary"

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


def _build_circle(spec: dict) -> GmshMeshResult:
    """
    Circular disk, centred at (cx, cy) with given radius.

    Boundaries: wall

    Parameters
    ----------
    radius      : disk radius [m]
    cx, cy      : centre coordinates (default 0, 0)
    mesh_size   : target element size (default: radius / 15)
    """
    r  = float(spec.get("radius", 0.05))
    cx = float(spec.get("cx", 0.0))
    cy = float(spec.get("cy", 0.0))
    ms = float(spec.get("mesh_size", r / 15.0))

    _init_gmsh("circle")
    disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, lambda cx, cy, _t: "wall")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


def _build_annulus(spec: dict) -> GmshMeshResult:
    """
    Annular ring between inner radius r_in and outer radius r_out.

    Boundaries: inner_wall, outer_wall

    Parameters
    ----------
    r_in, r_out : inner/outer radii [m]
    cx, cy      : centre coordinates (default 0, 0)
    mesh_size   : target element size (default: (r_out - r_in) / 10)
    """
    r_in  = float(spec.get("r_in",  0.01))
    r_out = float(spec.get("r_out", 0.05))
    cx    = float(spec.get("cx", 0.0))
    cy    = float(spec.get("cy", 0.0))
    ms    = float(spec.get("mesh_size", (r_out - r_in) / 10.0))

    _init_gmsh("annulus")
    outer = gmsh.model.occ.addDisk(cx, cy, 0, r_out, r_out)
    inner = gmsh.model.occ.addDisk(cx, cy, 0, r_in,  r_in)
    gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    mid = (r_in + r_out) / 2
    def classify_annulus(_c_x, _c_y, ctag):
        # For circular arcs the center-of-mass is the circle's centre, so we
        # distinguish inner/outer by measuring the curve's bounding-box radius.
        bb = gmsh.model.occ.getBoundingBox(1, ctag)
        curve_r = (bb[3] - bb[0]) / 2   # half the x-extent
        return "inner_wall" if curve_r < mid else "outer_wall"

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, classify_annulus)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


def _build_hollow_rectangle(spec: dict) -> GmshMeshResult:
    """
    Rectangle with a rectangular hole cut out of the interior.

    Boundaries: left, right, bottom, top, hole_wall

    Parameters
    ----------
    Lx, Ly          : outer dimensions [m]
    hole_w, hole_h  : hole dimensions [m]
    hole_cx, hole_cy: centre of the hole (default: Lx/2, Ly/2)
    mesh_size       : target element size
    """
    Lx      = float(spec.get("Lx",      0.1))
    Ly      = float(spec.get("Ly",      0.08))
    hole_w  = float(spec.get("hole_w",  Lx * 0.3))
    hole_h  = float(spec.get("hole_h",  Ly * 0.3))
    hole_cx = float(spec.get("hole_cx", Lx / 2))
    hole_cy = float(spec.get("hole_cy", Ly / 2))
    ms      = float(spec.get("mesh_size", min(Lx, Ly) / 20.0))

    _init_gmsh("hollow_rectangle")
    outer = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    inner = gmsh.model.occ.addRectangle(
        hole_cx - hole_w / 2, hole_cy - hole_h / 2, 0, hole_w, hole_h
    )
    gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    tol = min(Lx, Ly) * 1e-3
    def classify(cx_c, cy_c, _ctag):
        if   abs(cx_c)      < tol:  return "left"
        elif abs(cx_c - Lx) < tol:  return "right"
        elif abs(cy_c)      < tol:  return "bottom"
        elif abs(cy_c - Ly) < tol:  return "top"
        return "hole_wall"

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


def _build_t_shape(spec: dict) -> GmshMeshResult:
    """
    T-shaped cross-section (flange + web).

         ←───── flange_w ──────→
         ┌──────────────────────┐
         │       flange         │  flange_h
    ─────┘                      └─────
    web_h │       web           │
         └───────────────────────┘
               ←── web_w ──→

    Boundaries: left, right, top (flange), bottom (web),
                inner_left, inner_right (steps between flange and web)
    """
    flange_w = float(spec.get("flange_w", 0.1))
    flange_h = float(spec.get("flange_h", 0.02))
    web_w    = float(spec.get("web_w",    0.04))
    web_h    = float(spec.get("web_h",    0.08))
    ms       = float(spec.get("mesh_size", min(web_w, flange_h) / 8.0))

    _init_gmsh("t_shape")
    flange = gmsh.model.occ.addRectangle(0, web_h, 0, flange_w, flange_h)
    web    = gmsh.model.occ.addRectangle(
        (flange_w - web_w) / 2, 0, 0, web_w, web_h
    )
    gmsh.model.occ.fuse([(2, flange)], [(2, web)])
    gmsh.model.occ.synchronize()

    total_h   = web_h + flange_h
    web_left  = (flange_w - web_w) / 2
    web_right = (flange_w + web_w) / 2

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    tol = min(web_w, flange_h, web_h) * 1e-2
    def classify(cx_c, cy_c, _ctag):
        if   abs(cx_c)              < tol:                         return "left"
        elif abs(cx_c - flange_w)   < tol:                         return "right"
        elif abs(cy_c - total_h)    < tol:                         return "top"
        elif abs(cy_c)              < tol:                         return "bottom"
        elif abs(cy_c - web_h)      < tol and cx_c < web_left  + tol: return "inner_left"
        elif abs(cy_c - web_h)      < tol and cx_c > web_right - tol: return "inner_right"
        elif abs(cx_c - web_left)   < tol:                         return "web_left"
        elif abs(cx_c - web_right)  < tol:                         return "web_right"
        return "boundary"

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


def _build_stepped_notch(spec: dict) -> GmshMeshResult:
    """
    Rectangular bar with a single step/notch on the right side.
    Useful for studying stress/thermal concentrations.

    ┌──────────────────┐
    │                  │ Ly
    │         ┌────────┘
    │         │ step_h
    └─────────┘
    ← step_x →
    ←───── Lx ────────→

    Boundaries: left, bottom, right_lower, step_face, right_upper, top
    """
    Lx     = float(spec.get("Lx",     0.1))
    Ly     = float(spec.get("Ly",     0.06))
    step_x = float(spec.get("step_x", Lx * 0.6))
    step_h = float(spec.get("step_h", Ly * 0.4))
    ms     = float(spec.get("mesh_size", min(Lx, Ly) / 20.0))

    _init_gmsh("stepped_notch")
    full  = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    notch = gmsh.model.occ.addRectangle(step_x, 0, 0, Lx - step_x, step_h)
    gmsh.model.occ.cut([(2, full)], [(2, notch)])
    gmsh.model.occ.synchronize()

    surfs = gmsh.model.occ.getEntities(2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=1, name="domain")

    tol = min(Lx, Ly) * 1e-3
    def classify(cx_c, cy_c, _ctag):
        if   abs(cx_c)          < tol:                           return "left"
        elif abs(cy_c)          < tol and cx_c < step_x + tol:  return "bottom"
        elif abs(cx_c - Lx)     < tol and cy_c > step_h - tol:  return "right_upper"
        elif abs(cx_c - Lx)     < tol:                           return "right_lower"
        elif abs(cy_c - step_h) < tol:                           return "step_face"
        elif abs(cx_c - step_x) < tol:                           return "step_riser"
        elif abs(cy_c - Ly)     < tol:                           return "top"
        return "boundary"

    curves = gmsh.model.getBoundary(surfs, oriented=False)
    bn = _register_boundary_groups(curves, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(2)


# ─── 3D Geometries ────────────────────────────────────────────────────────────

def _build_box(spec: dict) -> GmshMeshResult:
    """
    Axis-aligned box [0,Lx]×[0,Ly]×[0,Lz].

    Boundaries: left (x=0), right (x=Lx), front (y=0), back (y=Ly),
                bottom (z=0), top (z=Lz)
    """
    Lx = float(spec.get("Lx", 1.0))
    Ly = float(spec.get("Ly", 1.0))
    Lz = float(spec.get("Lz", 1.0))
    ms = float(spec.get("mesh_size", min(Lx, Ly, Lz) / 8.0))

    _init_gmsh("box")
    gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.occ.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1, name="domain")

    tol = min(Lx, Ly, Lz) * 1e-3
    def classify(cx, cy, cz, _stag):
        if   abs(cx)      < tol:  return "left"
        elif abs(cx - Lx) < tol:  return "right"
        elif abs(cy)      < tol:  return "front"
        elif abs(cy - Ly) < tol:  return "back"
        elif abs(cz)      < tol:  return "bottom"
        elif abs(cz - Lz) < tol:  return "top"
        return "face"

    surfs = gmsh.model.getBoundary(vols, oriented=False)
    bn = _register_surface_groups(surfs, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(3)


def _build_cylinder(spec: dict) -> GmshMeshResult:
    """
    Circular cylinder, axis along z, radius r, height h.

    Boundaries: lateral_wall, bottom_face (z=0), top_face (z=h)
    """
    r  = float(spec.get("radius", 0.05))
    h  = float(spec.get("height", 0.1))
    ms = float(spec.get("mesh_size", min(r, h) / 8.0))

    _init_gmsh("cylinder")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, h, r)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.occ.getEntities(3)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], tag=1, name="domain")

    tol = min(r, h) * 1e-3
    def classify(cx, cy, cz, _stag):
        if   abs(cz)     < tol:  return "bottom_face"
        elif abs(cz - h) < tol:  return "top_face"
        return "lateral_wall"

    surfs = gmsh.model.getBoundary(vols, oriented=False)
    bn = _register_surface_groups(surfs, classify)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms)
    return _to_mesh(3)


# ─── Registry ─────────────────────────────────────────────────────────────────

GEOMETRY_REGISTRY: dict[str, object] = {
    "rectangle":        _build_rectangle,
    "l_shape":          _build_l_shape,
    "circle":           _build_circle,
    "annulus":          _build_annulus,
    "hollow_rectangle": _build_hollow_rectangle,
    "t_shape":          _build_t_shape,
    "stepped_notch":    _build_stepped_notch,
    "box":              _build_box,
    "cylinder":         _build_cylinder,
}


def build_gmsh_geometry(spec: dict) -> GmshMeshResult:
    """
    Build a mesh from a geometry specification dict.

    Parameters
    ----------
    spec : dict with at minimum {"type": "<geometry_name>", ...}

    Returns
    -------
    GmshMeshResult with mesh, facet_tags, cell_tags, boundary_names, dim.

    Example
    -------
    result = build_gmsh_geometry({
        "type": "l_shape",
        "Lx": 0.1, "Ly": 0.1,
        "cut_x": 0.05, "cut_y": 0.05,
        "mesh_size": 0.004,
    })
    # Available boundaries: left, bottom_left, bottom_right, right, inner_h, inner_v, top
    """
    if not GMSH_AVAILABLE:
        raise RuntimeError(
            "gmsh and/or dolfinx.io.gmsh are not available. "
            "Run inside the fenics-runner container."
        )

    geo_type = spec.get("type")
    if not geo_type:
        raise ValueError("Geometry spec must contain 'type' key.")

    builder = GEOMETRY_REGISTRY.get(geo_type)
    if builder is None:
        available = ", ".join(GEOMETRY_REGISTRY.keys())
        raise ValueError(
            f"Unknown geometry type '{geo_type}'. Available: {available}"
        )

    log.info("Building Gmsh geometry: %s  spec=%s", geo_type, spec)
    result = builder(spec)
    log.info(
        "Mesh built: %d cells, boundaries=%s",
        result.mesh.topology.index_map(result.dim).size_local,
        list(result.boundary_names.keys()),
    )
    return result
