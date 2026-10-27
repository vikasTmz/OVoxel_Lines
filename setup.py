"""CPU lines install by default when no CUDA/ROCm toolchain is available.

Set BUILD_TARGET=cpu for a NumPy-only install, or cuda/rocm for the original
mesh extension (install torch first and use --no-build-isolation).
"""
import os
from setuptools import find_packages, setup


def extension_config():
    target = os.environ.get("BUILD_TARGET", "auto").lower()
    if target not in {"auto", "cpu", "cuda", "rocm"}:
        raise ValueError("BUILD_TARGET must be auto, cpu, cuda or rocm")
    if target == "cpu":
        return {}
    try:
        from torch.utils.cpp_extension import (
            CUDAExtension, BuildExtension, CUDA_HOME, ROCM_HOME, IS_HIP_EXTENSION,
        )
    except ImportError:
        if target == "auto":
            return {}
        raise RuntimeError("Install torch before building the GPU extension") from None
    if target == "auto" and CUDA_HOME is None and ROCM_HOME is None:
        return {}
    is_hip = IS_HIP_EXTENSION if target == "auto" else target == "rocm"
    flags = ([f"--offload-arch={arch}" for arch in os.getenv("GPU_ARCHS", "native").split(";")]
             if is_hip else [])
    extension = CUDAExtension(
        name="o_voxel._C",
        sources=[
            "src/hash/hash.cu",
            "src/convert/flexible_dual_grid.cpp",
            "src/convert/volumetic_attr.cpp",
            "src/serialize/api.cu", "src/serialize/hilbert.cu", "src/serialize/z_order.cu",
            "src/io/svo.cpp", "src/io/filter_parent.cpp", "src/io/filter_neighbor.cpp",
            "src/rasterize/rasterize.cu", "src/ext.cpp",
        ],
        include_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/eigen")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"],
                            "nvcc": ["-O3", "-std=c++17"] + flags},
    )
    return {"ext_modules": [extension], "cmdclass": {"build_ext": BuildExtension}}


setup(name="o_voxel", packages=find_packages(include=["o_voxel", "o_voxel.*"]),
      **extension_config())
