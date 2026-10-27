"""Curve conversion needs only NumPy; mesh conversion uses the original extension."""
from importlib import import_module
from .line_dual_grid import (
    LineDualGrid, lines_to_flexible_dual_grid, flexible_dual_grid_to_lines,
)

_MESH_FUNCTIONS = {
    "mesh_to_flexible_dual_grid": ".flexible_dual_grid",
    "flexible_dual_grid_to_mesh": ".flexible_dual_grid",
    "textured_mesh_to_volumetric_attr": ".volumetic_attr",
    "blender_dump_to_volumetric_attr": ".volumetic_attr",
}
__all__ = ["LineDualGrid", "lines_to_flexible_dual_grid",
           "flexible_dual_grid_to_lines", *_MESH_FUNCTIONS]


def __getattr__(name):
    if name in _MESH_FUNCTIONS:
        value = getattr(import_module(_MESH_FUNCTIONS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
