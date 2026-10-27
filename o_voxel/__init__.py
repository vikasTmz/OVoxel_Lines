"""O-Voxel geometry tools, with optional GPU mesh dependencies loaded on demand."""
from importlib import import_module

__all__ = ["convert", "io", "postprocess", "rasterize", "serialize"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
