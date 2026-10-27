"""Reconstruct lines from a saved curve O-Voxel archive, without the source CAD."""
import argparse
from pathlib import Path
import numpy as np
from o_voxel.convert import LineDualGrid, flexible_dual_grid_to_lines
from o_voxel.convert.line_metrics import write_line_obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="Output .obj or .npz")
    args = parser.parse_args()
    if args.output.suffix.lower() not in (".obj", ".npz"):
        parser.error("Output extension must be .obj or .npz")
    vertices, segments = flexible_dual_grid_to_lines(LineDualGrid.load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".obj":
        write_line_obj(args.output, vertices, segments)
    else:
        np.savez_compressed(args.output, vertices=vertices, segments=segments)
    print(f"Reconstructed {len(vertices)} vertices and {len(segments)} lines: {args.output}")


if __name__ == "__main__":
    main()
