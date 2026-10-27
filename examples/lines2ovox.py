"""STEP boundaries (or indexed segments in NPZ) -> curve O-Voxels -> lines.

Run from the repository root after installing the package:
    python examples/lines2ovox.py --input assets/helmet/24380_528e79de.step
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from o_voxel.convert import LineDualGrid, lines_to_flexible_dual_grid, flexible_dual_grid_to_lines
from o_voxel.convert.cad_lines import load_step_lines
from o_voxel.convert.line_metrics import graph_signature, sampled_line_distance, write_line_obj
from o_voxel.convert.line_visualize import write_line_grid_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("assets/helmet/24380_528e79de.step"))
    parser.add_argument("--output", type=Path, default=Path("outputs/helmet_lines"))
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--deflection", type=float, default=.05, help="CAD chord-error target in import units (normally mm)")
    parser.add_argument("--nurbs-only", action="store_true", help="Select B-spline edges only, omit analytic edges")
    parser.add_argument("--line-weight", type=float, default=1.)
    parser.add_argument("--boundary-weight", type=float, default=0.)
    parser.add_argument("--regularization-weight", type=float, default=.01)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--metrics-samples", type=int, default=3, help="Samples per edge for distance estimates; 0 disables")
    args = parser.parse_args()
    if args.metrics_samples == 1 or args.metrics_samples < 0:
        parser.error("--metrics-samples must be 0 or >=2")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.input.suffix.lower() in (".step", ".stp"):
        vertices, segments, edge_ids, metadata = load_step_lines(
            args.input, deflection=args.deflection, nurbs_only=args.nurbs_only)
    elif args.input.suffix.lower() == ".npz":
        with np.load(args.input, allow_pickle=False) as data:
            vertices, segments = data["vertices"], data["segments"]
        edge_ids, metadata = np.arange(len(segments)), {"source": str(args.input)}
    else:
        parser.error("Input must be a STEP file or NPZ with vertices and segments")
    metadata["extraction_seconds"] = time.perf_counter() - started
    print(f"Input: {len(vertices)} vertices, {len(segments)} segments")
    np.savez_compressed(args.output / "input_lines.npz", vertices=vertices, segments=segments, cad_edge_ids=edge_ids)
    write_line_obj(args.output / "input_lines.obj", vertices, segments)
    started = time.perf_counter()
    grid = lines_to_flexible_dual_grid(
        vertices, segments, grid_size=args.resolution, line_weight=args.line_weight,
        boundary_weight=args.boundary_weight, regularization_weight=args.regularization_weight, timing=True)
    elapsed = time.perf_counter() - started
    grid.save(args.output / "line_ovoxels.npz")
    # Reconstruct from the persisted representation, not from the input lines.
    loaded = LineDualGrid.load(args.output / "line_ovoxels.npz")
    recovered, links = flexible_dual_grid_to_lines(loaded)
    np.savez_compressed(args.output / "reconstructed_lines.npz", vertices=recovered, segments=links)
    write_line_obj(args.output / "reconstructed_lines.obj", recovered, links)
    source_topology, output_topology = graph_signature(vertices, segments), graph_signature(recovered, links)
    if source_topology != output_topology:
        raise RuntimeError(f"Topology mismatch: {source_topology} vs {output_topology}")
    report = dict(
        source=metadata, resolution=grid.grid_size.tolist(), voxel_size=grid.voxel_size.tolist(),
        input_vertices=len(vertices), input_segments=len(segments), active_voxels=len(grid.coords),
        dual_vertices=len(grid.dual_vertices), anchors=len(grid.anchor_vertices), output_segments=len(links),
        cells_with_multiple_duals=int(np.sum(np.bincount(grid.dual_voxel_indices) > 1)),
        conversion_seconds=elapsed, input_topology=source_topology, output_topology=output_topology,
        weights=dict(line=args.line_weight, boundary=args.boundary_weight, regularization=args.regularization_weight),
    )
    if args.metrics_samples:
        report["sampled_input_to_output"] = sampled_line_distance(vertices, segments, recovered, links, args.metrics_samples)
        report["sampled_output_to_input"] = sampled_line_distance(recovered, links, vertices, segments, args.metrics_samples)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.no_html:
        write_line_grid_html(args.output / "viewer.html", loaded, input_vertices=vertices, input_segments=segments)
    print(json.dumps(report, indent=2))
    print(f"Wrote outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
