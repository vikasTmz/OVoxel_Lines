"""Small NumPy-only diagnostics for line graphs."""
import numpy as np


def graph_signature(vertices, segments):
    """Topological invariants, ignoring unreferenced and degree-two vertices."""
    segments = np.unique(np.sort(segments, axis=1), axis=0)
    used = np.unique(segments)
    parent = np.arange(len(vertices))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in segments:
        parent[root(a)] = root(b)
    components = len({root(i) for i in used})
    degree = np.bincount(segments.ravel(), minlength=len(vertices))[used]
    junctions, counts = np.unique(degree[degree > 2], return_counts=True)
    return dict(components=components, cycle_rank=len(segments) - len(used) + components,
                endpoints=int(np.sum(degree == 1)),
                junction_degrees={str(d): int(c) for d, c in zip(junctions, counts)})


def sampled_line_distance(source_vertices, source_edges, target_vertices, target_edges, samples=3):
    """Directed point-to-segment distance, sampling each source segment uniformly.

    This is a sampled estimate, not a certified Hausdorff distance. Means are
    sample-weighted, so subdivision density affects them. Work is O(samples *
    source_edges * target_edges), with bounded-size temporary arrays.
    """
    if not isinstance(samples, int) or samples < 2:
        raise ValueError("samples must be an integer >=2")
    if not len(source_edges) or not len(target_edges):
        raise ValueError("Distance requires two nonempty line graphs")
    source = source_vertices[source_edges]
    t = np.linspace(0, 1, samples)[None, :, None]
    points = (source[:, :1] * (1 - t) + source[:, 1:] * t).reshape(-1, 3)
    target = target_vertices[target_edges]
    starts, directions = target[:, 0], target[:, 1] - target[:, 0]
    norm2 = np.einsum("ij,ij->i", directions, directions)
    distance2 = []
    for offset in range(0, len(points), 128):
        batch = points[offset:offset + 128]
        best = np.full(len(batch), np.inf)
        for j in range(0, len(starts), 2048):
            delta = batch[:, None] - starts[None, j:j + 2048]
            d, n = directions[j:j + 2048], norm2[j:j + 2048]
            parameter = np.divide(np.einsum("bni,ni->bn", delta, d), n,
                                  out=np.zeros(delta.shape[:2]), where=n > 0)
            delta -= np.clip(parameter, 0, 1)[..., None] * d
            best = np.minimum(best, np.einsum("bni,bni->bn", delta, delta).min(axis=1))
        distance2.extend(best.tolist())
    distance2 = np.asarray(distance2)
    return dict(samples=len(points), mean=float(np.sqrt(distance2).mean()),
                rms=float(np.sqrt(distance2.mean())), maximum=float(np.sqrt(distance2.max())))


def write_line_obj(path, vertices, segments):
    """Wavefront OBJ with native line elements (l), not degenerate triangles."""
    with open(path, "w") as stream:
        stream.write("# Curve O-Voxel line graph; world coordinates\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b in segments:
            stream.write(f"l {a + 1} {b + 1}\n")
