"""Gmsh-based geometry builders for FEniCSx simulations."""
from .gmsh_geometries import build_gmsh_geometry, GEOMETRY_REGISTRY, GmshMeshResult

__all__ = ["build_gmsh_geometry", "GEOMETRY_REGISTRY", "GmshMeshResult"]
