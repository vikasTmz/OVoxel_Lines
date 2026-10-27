"""Export a standalone interactive HTML viewer; no plotting dependency or CDN."""
import json
from pathlib import Path
import numpy as np
from .line_dual_grid import flexible_dual_grid_to_lines


def write_line_grid_html(path, grid, *, input_vertices=None, input_segments=None):
    vertices, segments = flexible_dual_grid_to_lines(grid)
    payload = dict(
        vertices=vertices.tolist(), segments=segments.tolist(),
        dual_count=len(grid.dual_vertices), coords=grid.coords.tolist(),
        membership=grid.dual_voxel_indices.tolist(), local=grid.dual_vertices.tolist(),
        flags=grid.face_flags.tolist(), crossings=grid.link_points.tolist(),
        axes=grid.link_axes.tolist(), aabb=grid.aabb.tolist(), size=grid.grid_size.tolist(),
        input_vertices=[] if input_vertices is None else np.asarray(input_vertices).tolist(),
        input_segments=[] if input_segments is None else np.asarray(input_segments).tolist(),
    )
    template = Path(__file__).with_name("line_viewer.html").read_text()
    Path(path).write_text(template.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":"))))
