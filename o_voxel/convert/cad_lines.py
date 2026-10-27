"""Tessellate STEP B-rep boundary curves, not the edges of a triangle mesh."""
from pathlib import Path
import numpy as np


def load_step_lines(path, *, deflection=0.05, nurbs_only=False):
    """Return vertices, segments, segment CAD-edge IDs, and extraction metadata.

    OpenCascade reads trimmed, located B-rep edges, including rational splines.
    Each topological edge is sampled once even when shared by several faces.
    Shared CAD vertices get shared IDs; coincident but independent bodies are
    not welded. Analytic boundaries are included unless nurbs_only=True.
    deflection is the OCCT sampler's chord-error target in imported model units
    (STEPControl normally converts lengths to millimeters).
    """
    if not np.isfinite(deflection) or deflection <= 0:
        raise ValueError("deflection must be positive and finite")
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.TopExp import TopExp
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopoDS import TopoDS
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GCPnts import GCPnts_QuasiUniformDeflection
        from OCP.GeomAbs import GeomAbs_BSplineCurve
        try:
            from OCP.TopTools import TopTools_IndexedMapOfShape as ShapeMap
        except ImportError:  # OCCT 8 moved templated collections.
            from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeMap
    except ImportError as exc:
        raise ImportError("STEP input requires: pip install 'o_voxel[cad]' (Python >=3.11)") from exc
    edge_cast = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"OpenCascade could not read STEP file: {path}")
    if reader.TransferRoots() == 0:
        raise ValueError("STEP file has no transferable roots")
    shape = reader.OneShape()
    edges, cad_vertices = ShapeMap(), ShapeMap()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edges)
    vertices, segments, edge_ids = [], [], []
    vertex_ids, curve_types = {}, {}
    skipped, selected = 0, 0

    def xyz(point):
        return np.array([point.X(), point.Y(), point.Z()])

    def endpoint(vertex, fallback):
        if vertex.IsNull():
            vertices.append(fallback)
            return len(vertices) - 1
        index = cad_vertices.Add(vertex)
        if index not in vertex_ids:
            vertex_ids[index] = len(vertices)
            vertices.append(xyz(BRep_Tool.Pnt_s(vertex)))
        return vertex_ids[index]

    for index in range(1, edges.Extent() + 1):
        edge = edge_cast(edges.FindKey(index))
        if BRep_Tool.Degenerated_s(edge):
            skipped += 1
            continue
        curve = BRepAdaptor_Curve(edge)
        curve_type = curve.GetType()
        if nurbs_only and curve_type != GeomAbs_BSplineCurve:
            continue
        kind = str(curve_type).split(".")[-1]
        curve_types[kind] = curve_types.get(kind, 0) + 1
        selected += 1
        sampler = GCPnts_QuasiUniformDeflection(curve, float(deflection))
        if not sampler.IsDone() or sampler.NbPoints() < 2:
            raise RuntimeError(f"Cannot sample CAD edge {index}; try a smaller deflection")
        parameters = [sampler.Parameter(i) for i in range(1, sampler.NbPoints() + 1)]
        first, last = curve.FirstParameter(), curve.LastParameter()
        start = endpoint(TopExp.FirstVertex_s(edge), xyz(curve.Value(first)))
        end = endpoint(TopExp.LastVertex_s(edge), xyz(curve.Value(last)))
        # Even a tiny closed edge must have a polygonal cycle, not one point.
        if start == end:
            parameters = sorted(set(parameters + np.linspace(first, last, 5).tolist()))
        chain = [start]
        for u in parameters[1:-1]:
            point = xyz(curve.Value(u))
            if np.array_equal(point, vertices[chain[-1]]):
                continue
            chain.append(len(vertices))
            vertices.append(point)
        chain.append(end)
        for a, b in zip(chain[:-1], chain[1:]):
            if np.array_equal(vertices[a], vertices[b]):
                continue
            segments.append((a, b))
            edge_ids.append(index)
    if not segments:
        raise ValueError("No nondegenerate boundary curves were selected")
    metadata = {
        "source": str(path), "unique_cad_edges": edges.Extent(),
        "selected_cad_edges": selected, "degenerate_cad_edges_skipped": skipped,
        "curve_types": curve_types, "deflection": deflection,
        "nurbs_only": nurbs_only, "length_units": "OpenCascade import units (normally mm)",
    }
    return (np.asarray(vertices, dtype=float), np.asarray(segments, dtype=np.int64),
            np.asarray(edge_ids, dtype=np.int64), metadata)
