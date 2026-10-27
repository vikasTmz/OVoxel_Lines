"""Sparse, topology-aware flexible dual grids for embedded line graphs.

The rank-two line QEF uses I - t t^T, not a surface-normal outer product.
All geometry is float64 on the CPU. No compiled extension is required.
"""
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
import time

import numpy as np

__all__ = ["LineDualGrid", "lines_to_flexible_dual_grid",
           "flexible_dual_grid_to_lines"]


def _array(value, dtype=None):
    # Accept torch inputs without importing torch or hiding a CPU transfer.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _integer_array(value, name, shape=None):
    a = _array(value)
    if a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers")
    if a.size and (np.min(a) < 0 or np.max(a) > np.iinfo(np.int64).max):
        raise ValueError(f"{name} has an out-of-range index")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return a.astype(np.int64)


@dataclass
class LineDualGrid:
    """Version 1 curve O-Voxel representation (not the mesh .vxz schema).

    coords: unique occupied cells (N,3); dual_voxel_indices: (K,) rows in coords.
    dual_vertices: (K,3) cell-local coordinates in [0,1]. Multiple duals in a
        cell preserve disconnected strands, branches, and unresolved cycles.
    face_flags: (K,6), ordered -x,+x,-y,+y,-z,+z; faces touched by the local
        curve component, including tangencies and coplanar contacts.
    anchor_vertices: (A,3) world-space open endpoints and graph junctions.
    edges: (M,2) indices into [world dual vertices, anchor_vertices].
    link_points: (M,3) world-space original connection points, for inspection.
    link_axes: (M,3) which grid planes a dual-to-dual connection crosses;
        simultaneous edge/corner crossings have two/three true entries.
    aabb: (2,3) world bounds; grid_size: (3,) number of cells.

    Reconstruction needs only positions, membership, edges, and grid metadata;
    link_points, link_axes and face_flags are diagnostic intersection data.
    """
    coords: np.ndarray
    dual_voxel_indices: np.ndarray
    dual_vertices: np.ndarray
    face_flags: np.ndarray
    anchor_vertices: np.ndarray
    edges: np.ndarray
    link_points: np.ndarray
    link_axes: np.ndarray
    aabb: np.ndarray
    grid_size: np.ndarray

    @property
    def voxel_size(self):
        return (self.aabb[1] - self.aabb[0]) / self.grid_size

    @property
    def world_dual_vertices(self):
        return (self.aabb[0] +
                (self.coords[self.dual_voxel_indices] + self.dual_vertices) * self.voxel_size)

    def validate(self):
        """Reject malformed archives/edits before indexing or reconstruction."""
        n, k, a, m = (len(self.coords), len(self.dual_vertices),
                      len(self.anchor_vertices), len(self.edges))
        for name, shape in [("coords", (n, 3)), ("grid_size", (3,)),
                            ("dual_voxel_indices", (k,)), ("edges", (m, 2))]:
            _integer_array(getattr(self, name), name, shape)
        for name, shape in [("dual_vertices", (k, 3)), ("anchor_vertices", (a, 3)),
                            ("link_points", (m, 3)), ("aabb", (2, 3))]:
            val = np.asarray(getattr(self, name))
            if val.shape != shape or not np.isfinite(val).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        for name, shape in [("face_flags", (k, 6)), ("link_axes", (m, 3))]:
            val = np.asarray(getattr(self, name))
            if val.shape != shape or val.dtype != np.bool_:
                raise ValueError(f"{name} must be boolean with shape {shape}")
        if np.any(self.grid_size <= 0) or np.any(self.aabb[1] <= self.aabb[0]):
            raise ValueError("grid_size and aabb extents must be positive")
        if np.any(self.coords >= self.grid_size) or len(np.unique(self.coords, axis=0)) != n:
            raise ValueError("coords must be unique and within the grid")
        if k and (self.dual_voxel_indices.max() >= n):
            raise ValueError("dual_voxel_indices out of range")
        if np.any(self.dual_vertices < 0) or np.any(self.dual_vertices > 1):
            raise ValueError("dual_vertices must be cell-local coordinates in [0,1]")
        if m and (self.edges.max() >= k + a or np.any(self.edges[:, 0] == self.edges[:, 1])):
            raise ValueError("edges contain an invalid index or self-loop")
        if np.any(self.anchor_vertices < self.aabb[0]) or np.any(self.anchor_vertices > self.aabb[1]):
            raise ValueError("anchors must lie inside aabb")
        return self

    def save(self, path):
        """Write a compressed NumPy archive, without pickle or original lines."""
        self.validate()
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, format_version=np.array(1),
                                **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if data["format_version"].shape != () or int(data["format_version"]) != 1:
                raise ValueError("Unsupported line O-Voxel format version")
            obj = cls(**{f.name: data[f.name] for f in fields(cls)})
        return obj.validate()


def _grid(vertices, grid_size, voxel_size, aabb):
    if (grid_size is None) == (voxel_size is None):
        raise ValueError("Provide exactly one of grid_size or voxel_size")
    if grid_size is not None:
        size = _integer_array(grid_size, "grid_size")
        size = np.repeat(size, 3) if size.ndim == 0 else size
        if size.shape != (3,) or np.any(size <= 0):
            raise ValueError("grid_size must be a positive integer or three positive integers")
    else:
        step = _array(voxel_size, np.float64)
        step = np.repeat(step, 3) if step.ndim == 0 else step
        if step.shape != (3,) or not np.isfinite(step).all() or np.any(step <= 0):
            raise ValueError("voxel_size must be positive and finite")
    if aabb is None:
        if not len(vertices):
            raise ValueError("aabb is required for empty input")
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        center = (lo + hi) / 2
        extent = hi - lo
        # A cube avoids singular boxes for planar/axis-aligned curves.
        span = max(float(extent.max()), 1e-6)
        if grid_size is not None:
            extent = np.full(3, span * 1.02)
        else:
            size = np.maximum(1, np.ceil(extent / step).astype(np.int64) + 2)
            extent = size * step
        box = np.array([center - extent / 2, center + extent / 2])
    else:
        box = _array(aabb, np.float64).copy()
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] <= box[0]):
        raise ValueError("aabb must be finite with shape (2,3) and positive extents")
    if voxel_size is not None:
        ratio = (box[1] - box[0]) / step
        if not np.allclose(ratio, np.rint(ratio), rtol=1e-10, atol=1e-10):
            raise ValueError("Explicit aabb extent must be an integer multiple of voxel_size")
        size = np.rint(ratio).astype(np.int64)
        if np.any(size <= 0):
            raise ValueError("aabb must contain at least one voxel")
    if np.any(vertices < box[0]) or np.any(vertices > box[1]):
        raise ValueError("Input lies outside aabb; clip the line graph explicitly first")
    return size, box, (box[1] - box[0]) / size


def _box_qef(matrix, rhs, lower, upper):
    """Exact 3-variable convex box QP via the 27 possible active sets.

    Coordinates are centered at the mass point. Least squares picks the
    closest-to-mass minimizer when the QEF is singular (regularization=0).
    Clipping an unconstrained solution alone would not solve this problem.
    """
    x = np.linalg.lstsq(matrix, rhs, rcond=1e-12)[0]
    if np.all(x >= lower) and np.all(x <= upper):
        return x
    best, best_key = None, (np.inf, np.inf)
    for state in product((-1, 0, 1), repeat=3):
        state = np.asarray(state)
        free, fixed = np.flatnonzero(state == 0), np.flatnonzero(state != 0)
        candidate = np.where(state < 0, lower, np.where(state > 0, upper, 0.))
        if len(free):
            candidate[free] = np.linalg.lstsq(
                matrix[np.ix_(free, free)],
                rhs[free] - matrix[np.ix_(free, fixed)] @ candidate[fixed], rcond=1e-12)[0]
        if np.any(candidate < lower - 1e-12) or np.any(candidate > upper + 1e-12):
            continue
        candidate = np.clip(candidate, lower, upper)
        cost = candidate @ matrix @ candidate - 2 * rhs @ candidate
        key = (cost, candidate @ candidate)
        if key < best_key:
            best, best_key = candidate, key
    if best is None:
        raise RuntimeError("No feasible QEF solution")
    return best


def _solve_component(points, piece_nodes, members, coord, box, step,
                     degree, line_weight, boundary_weight, regularization_weight):
    origin = box[0] + coord * step
    scale = float(step.min())
    local = (points[piece_nodes[members]] - origin) / scale
    delta = local[:, 1] - local[:, 0]
    length = np.linalg.norm(delta, axis=1)
    tangent = delta / length[:, None]
    weights = length / length.sum()
    midpoint = local.mean(axis=1)
    mass = weights @ midpoint
    projector = np.eye(3)[None] - tangent[:, :, None] * tangent[:, None, :]
    matrix = line_weight * np.einsum("n,nij->ij", weights, projector)
    rhs = line_weight * np.einsum("n,nij,nj->i", weights, projector, midpoint - mass)
    boundary_ids = np.unique(piece_nodes[members])
    boundary_ids = boundary_ids[degree[boundary_ids] != 2]
    if len(boundary_ids) and boundary_weight:
        target = (points[boundary_ids] - origin) / scale
        matrix += boundary_weight * np.eye(3)
        rhs += boundary_weight * (target.mean(axis=0) - mass)
    matrix += regularization_weight * np.eye(3)
    solution = mass + _box_qef(matrix, rhs, -mass, step / scale - mass)
    return np.clip(solution * scale / step, 0, 1)


def lines_to_flexible_dual_grid(
    vertices, segments, *, grid_size=None, voxel_size=None, aabb=None,
    line_weight=1.0, boundary_weight=0.0, regularization_weight=1e-2,
    timing=False,
):
    """Convert an indexed line graph into a :class:`LineDualGrid`.

    vertices (V,3) are in world coordinates; segments (E,2) are vertex IDs.
    Shared IDs define connectivity; geometric coincidences are NOT welded.
    Duplicate undirected segments are removed; zero-length segments are
    rejected. Unreferenced vertices are ignored. Inputs may be NumPy arrays,
    lists, or torch tensors; output is always CPU NumPy (non-differentiable).

    Exactly one of grid_size / voxel_size is required. Bounds are a padded
    cube by default, or supplied explicitly. Out-of-bounds input is rejected.
    Traversal uses half-open cells, with the outer maximum in the last cell.
    Lines on grid planes belong to the positive-side cell. Corner crossings
    connect diagonal cells directly, without artificial zero-length pieces.

    In-cell degree-two chains are contracted to one QEF dual. Endpoints and
    junctions are exact anchors. Contractions that erase a cycle or create
    parallel edges are refined to per-piece duals, retaining graph topology.
    """
    started = time.perf_counter()
    vertices = _array(vertices, np.float64)
    segments = _integer_array(segments, "segments")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite with shape (V,3)")
    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError("segments must have shape (E,2)")
    if segments.size and segments.max() >= len(vertices):
        raise ValueError("segment index out of range")
    weights = np.array([line_weight, boundary_weight, regularization_weight], dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or line_weight <= 0:
        raise ValueError("QEF weights must be finite/nonnegative and line_weight positive")
    segments = np.unique(np.sort(segments, axis=1), axis=0)
    used = np.unique(segments)
    size, box, step = _grid(vertices[used], grid_size, voxel_size, aabb)
    # Remap to discard unused vertices, while preserving distinct coincident IDs.
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    vertices, segments = vertices[used], remap[segments]
    points = list(vertices)
    piece_nodes, piece_coords = [], []
    grid_vertices = (vertices - box[0]) / step
    for va, vb in segments:
        start, end = grid_vertices[va], grid_vertices[vb]
        direction = end - start
        if not np.any(direction):
            raise ValueError("Zero-length segments are not curves; remove them before conversion")
        times = [0., 1.]
        for axis in range(3):
            if direction[axis] == 0:
                continue
            lo, hi = sorted((start[axis], end[axis]))
            planes = np.arange(max(1, int(np.floor(lo)) + 1),
                               min(int(size[axis]), int(np.ceil(hi))))
            times.extend(((planes - start[axis]) / direction[axis]).tolist())
        times = np.sort(times)
        # Merge floating-point representations of the same edge/corner event.
        times = times[np.r_[True, np.diff(times) > 32 * np.finfo(float).eps]]
        times[0], times[-1] = 0., 1.
        ids = [int(va)]
        for t in times[1:-1]:
            ids.append(len(points))
            points.append(vertices[va] + t * (vertices[vb] - vertices[va]))
        ids.append(int(vb))
        for j in range(len(times) - 1):
            midpoint = start + (times[j] + times[j + 1]) * .5 * direction
            coord = np.clip(np.floor(midpoint).astype(np.int64), 0, size - 1)
            piece_nodes.append((ids[j], ids[j + 1]))
            piece_coords.append(coord)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    piece_nodes = np.asarray(piece_nodes, dtype=np.int64).reshape(-1, 2)
    piece_coords = np.asarray(piece_coords, dtype=np.int64).reshape(-1, 3)
    incident = [[] for _ in points]
    for piece, (a, b) in enumerate(piece_nodes):
        incident[a].append(piece)
        incident[b].append(piece)
    degree = np.array([len(x) for x in incident])
    parent = np.arange(len(piece_nodes))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for pieces in incident:
        if len(pieces) == 2:
            a, b = pieces
            if np.array_equal(piece_coords[a], piece_coords[b]):
                parent[root(b)] = root(a)
    groups = {}
    for i in range(len(piece_nodes)):
        groups.setdefault(root(i), []).append(i)
    groups = list(groups.values())
    anchors = np.flatnonzero((degree != 2) & (degree > 0))

    def connections(groups):
        membership = np.empty(len(piece_nodes), dtype=np.int64)
        for group, members in enumerate(groups):
            membership[members] = group
        anchor_map = {v: len(groups) + i for i, v in enumerate(anchors)}
        links, nodes = [], []
        for node, pieces in enumerate(incident):
            if len(pieces) == 2:
                a, b = membership[pieces]
                if a != b:
                    links.append((int(a), int(b)))
                    nodes.append(node)
            elif pieces:
                for p in pieces:
                    links.append((int(membership[p]), anchor_map[node]))
                    nodes.append(node)
        return np.asarray(links, dtype=np.int64).reshape(-1, 2), nodes

    # A loop entirely in a cell has no ports; a loop visiting just two cells
    # can produce parallel links. Refine those groups rather than lose cycles.
    while True:
        edges, link_nodes = connections(groups)
        counts = np.bincount(edges.ravel(), minlength=len(groups) + len(anchors))
        refine = {i for i, g in enumerate(groups) if counts[i] == 0 and len(g) > 1}
        seen = set()
        for a, b in edges:
            pair = tuple(sorted((a, b)))
            if pair in seen:
                refine.update(i for i in pair if i < len(groups) and len(groups[i]) > 1)
            seen.add(pair)
        if not refine:
            break
        groups = [part for i, g in enumerate(groups)
                  for part in ([[p] for p in g] if i in refine else [g])]

    dual_coords = np.asarray([piece_coords[g[0]] for g in groups], dtype=np.int64).reshape(-1, 3)
    coords, voxel_indices = np.unique(dual_coords, axis=0, return_inverse=True)
    duals = np.asarray([
        _solve_component(points, piece_nodes, g, dual_coords[i], box, step, degree, *weights)
        for i, g in enumerate(groups)], dtype=float).reshape(-1, 3)
    link_points = points[link_nodes].reshape(-1, 3)
    flags = np.zeros((len(groups), 6), dtype=bool)
    link_axes = np.zeros((len(edges), 3), dtype=bool)
    for node, members in enumerate(groups):
        local = (points[piece_nodes[members].ravel()] - box[0]) / step - dual_coords[node]
        flags[node, 0::2] = np.isclose(local, 0, atol=1e-9, rtol=0).any(axis=0)
        flags[node, 1::2] = np.isclose(local, 1, atol=1e-9, rtol=0).any(axis=0)
    for i, (a, b) in enumerate(edges):
        if a < len(groups) and b < len(groups):
            link_axes[i] = dual_coords[a] != dual_coords[b]
    result = LineDualGrid(coords, voxel_indices, duals, flags, points[anchors],
                          edges, link_points, link_axes, box, size).validate()
    if timing:
        print(f"Line voxelization: {len(segments)} segments -> {len(coords)} cells, "
              f"{len(duals)} duals, {len(anchors)} anchors in {time.perf_counter()-started:.3f}s")
    return result


def flexible_dual_grid_to_lines(grid):
    """Return (world_vertices, segment_indices), independent of the source graph.

    Open endpoints/junctions are appended after dual vertices. This is a
    piecewise-linear approximation, not a reconstruction of NURBS parameters.
    """
    if not isinstance(grid, LineDualGrid):
        raise TypeError("grid must be a LineDualGrid")
    grid.validate()
    return np.concatenate((grid.world_dual_vertices, grid.anchor_vertices)), grid.edges.copy()
