import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from o_voxel.convert import (
    LineDualGrid, lines_to_flexible_dual_grid, flexible_dual_grid_to_lines,
)
from o_voxel.convert.line_dual_grid import _box_qef


def topology(vertices, edges):
    used = np.unique(edges)
    parent = list(range(len(vertices)))

    def root(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for a, b in edges:
        parent[root(a)] = root(b)
    components = len({root(i) for i in used})
    degree = np.bincount(edges.ravel(), minlength=len(vertices))[used]
    return (components, len(edges) - len(used) + components,
            sorted(degree[degree != 2].tolist()))


class CurveGridTests(unittest.TestCase):
    def convert(self, vertices, edges, size=4, **kwargs):
        return lines_to_flexible_dual_grid(vertices, edges, grid_size=size,
                                           aabb=[[0, 0, 0], [1, 1, 1]], **kwargs)

    def assert_topology(self, vertices, edges, size=4):
        grid = self.convert(vertices, edges, size)
        restored, links = flexible_dual_grid_to_lines(grid)
        self.assertEqual(topology(vertices, np.asarray(edges)), topology(restored, links))
        self.assertTrue(np.all(np.linalg.norm(restored[links[:, 1]] - restored[links[:, 0]], axis=1) > 0))
        return grid

    def test_axis_aligned_line_and_endpoints(self):
        v = np.array([[.1, .3, .4], [.9, .3, .4]])
        grid = self.assert_topology(v, [[0, 1]])
        self.assertEqual(len(grid.coords), 4)
        np.testing.assert_allclose(grid.world_dual_vertices[:, 1:], np.tile([.3, .4], (4, 1)))
        np.testing.assert_array_equal(grid.anchor_vertices, v)
        np.testing.assert_array_equal(grid.link_axes.sum(0), [3, 0, 0])

    def test_corner_and_edge_crossings(self):
        for end in ([1, 1, 1], [1, 1, 0]):
            grid = self.assert_topology([[0, 0, 0], end], [[0, 1]])
            self.assertEqual(len(grid.coords), 4)
            self.assertEqual(int(grid.link_axes.sum()), 3 * np.count_nonzero(end))

    def test_grid_plane_and_maximum_ownership(self):
        for y, expected in ((.5, 2), (1., 3)):
            grid = self.convert([[0, y, .5], [1, y, .5]], [[0, 1]])
            np.testing.assert_array_equal(grid.coords[:, 1], expected)
            self.assertEqual(len(grid.coords), 4)

    def test_segment_ending_at_grid_corner(self):
        self.assert_topology([[.1, .1, .1], [.5, .5, .5], [.9, .9, .9]], [[0, 1], [1, 2]])

    def test_short_segment_in_one_voxel(self):
        grid = self.assert_topology([[.11, .12, .13], [.14, .15, .16]], [[0, 1]])
        self.assertEqual(len(grid.coords), 1)
        self.assertEqual(len(grid.anchor_vertices), 2)
        self.assertEqual(len(grid.edges), 2)

    def test_unrelated_strands_in_same_cell(self):
        grid = self.assert_topology([[.1, .1, .1], [.2, .1, .1], [.1, .2, .1], [.2, .2, .1]],
                                    [[0, 1], [2, 3]])
        self.assertEqual(len(grid.coords), 1)
        self.assertEqual(len(grid.dual_vertices), 2)

    def test_geometric_crossing_is_not_a_junction(self):
        self.assert_topology([[.1, .5, .5], [.9, .5, .5], [.5, .1, .5], [.5, .9, .5]],
                             [[0, 1], [2, 3]])

    def test_exact_junction(self):
        v = [[.2, .2, .2], [.1, .2, .2], [.2, .1, .2], [.2, .2, .1]]
        grid = self.assert_topology(v, [[0, 1], [0, 2], [0, 3]])
        self.assertEqual(len(grid.anchor_vertices), 4)
        np.testing.assert_array_equal(grid.anchor_vertices, v)

    def test_closed_loop_one_cell(self):
        grid = self.assert_topology([[.1, .1, .1], [.2, .1, .1], [.15, .2, .1]],
                                    [[0, 1], [1, 2], [2, 0]])
        self.assertEqual(len(grid.coords), 1)
        self.assertGreaterEqual(len(grid.dual_vertices), 3)

    def test_closed_loop_two_cells(self):
        self.assert_topology([[.2, .1, .1], [.3, .1, .1], [.3, .2, .1], [.2, .2, .1]],
                             [[0, 1], [1, 2], [2, 3], [3, 0]])

    def test_loop_with_branch(self):
        self.assert_topology([[.1, .1, .1], [.2, .1, .1], [.15, .2, .1], [.05, .05, .1]],
                             [[0, 1], [1, 2], [2, 0], [0, 3]])

    def test_tangent_touch_plane(self):
        grid = self.assert_topology([[.1, .1, .2], [.25, .15, .2], [.1, .2, .2]], [[0, 1], [1, 2]])
        self.assertTrue(grid.face_flags[0, 1])

    def test_anisotropic_world_metric(self):
        a, b = np.array([1., 10., 100.]), np.array([9., 30., 180.])
        grid = lines_to_flexible_dual_grid([a, b], [[0, 1]], grid_size=[5, 4, 3],
                                           aabb=[[0, 0, 0], [10, 40, 200]])
        residual = np.cross(grid.world_dual_vertices - a, b - a)
        np.testing.assert_allclose(residual, 0, atol=1e-10)

    def test_qef_recovers_corner(self):
        grid = self.convert([[.1, .4, .2], [.4, .4, .2], [.4, .8, .2]],
                            [[0, 1], [1, 2]], size=1, regularization_weight=0)
        np.testing.assert_allclose(grid.world_dual_vertices, [[.4, .4, .2]], atol=1e-13)

    def test_qef_box_constraint_not_coordinate_clamping(self):
        matrix = np.array([[2., 1., 0.], [1., 2., 0.], [0., 0., 1.]])
        rhs = matrix @ [-1., .5, .5]
        solution = _box_qef(matrix, rhs, np.zeros(3), np.ones(3))
        np.testing.assert_allclose(solution, [0., 0., .5], atol=1e-13)
        clamped = np.clip(np.linalg.solve(matrix, rhs), 0, 1)
        cost = lambda x: x @ matrix @ x - 2 * rhs @ x
        self.assertLess(cost(solution), cost(clamped))

    def test_subdivision_invariance(self):
        a = self.convert([[.1, .3, .4], [.9, .3, .4]], [[0, 1]])
        v = np.column_stack([np.linspace(.1, .9, 35), np.full(35, .3), np.full(35, .4)])
        b = self.convert(v, np.column_stack([np.arange(34), np.arange(1, 35)]))
        np.testing.assert_allclose(a.world_dual_vertices, b.world_dual_vertices, atol=1e-13)

    def test_reversed_segments_and_duplicates(self):
        v = [[.1, .3, .4], [.9, .3, .4]]
        a = self.convert(v, [[0, 1]])
        b = self.convert(v, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(a.edges, b.edges)
        np.testing.assert_array_equal(a.dual_vertices, b.dual_vertices)

    def test_empty_and_serialization(self):
        for grid in (self.convert(np.empty((0, 3)), np.empty((0, 2), dtype=int)),
                     self.convert([[.1, .3, .4], [.9, .3, .4]], [[0, 1]])):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "lines.npz"
                grid.save(path)
                loaded = LineDualGrid.load(path)
                for name in grid.__dataclass_fields__:
                    np.testing.assert_array_equal(getattr(grid, name), getattr(loaded, name))
                flexible_dual_grid_to_lines(loaded)

    def test_auto_bounds_for_planar_curve_and_voxel_size(self):
        for kwargs in ({"grid_size": 1}, {"voxel_size": .2}):
            grid = lines_to_flexible_dual_grid([[0, 0, 0], [1, 0, 0]], [[0, 1]], **kwargs)
            self.assertTrue(np.all(grid.voxel_size > 0))
            self.assertEqual(topology(*flexible_dual_grid_to_lines(grid)), (1, 0, [1, 1]))

    def test_invalid_input(self):
        for v, e, kw in [
            ([[0, 0, 0], [0, 0, 0]], [[0, 1]], {}),
            ([[0, 0, 0]], [[-1, 0]], {}),
            ([[0, 0, 0]], [[0, 2]], {}),
            ([[0, 0, 0]], [[0., 0.]], {}),
            ([[0, 0, np.nan]], [[0, 0]], {}),
            ([[-.1, 0, 0], [1, 0, 0]], [[0, 1]], {}),
            ([[0, 0, 0], [1, 0, 0]], [[0, 1]], {"line_weight": -1}),
            ([[0, 0, 0], [1, 0, 0]], [[0, 1]], {"regularization_weight": np.nan}),
        ]:
            with self.subTest(v=v, e=e, kw=kw), self.assertRaises(ValueError):
                self.convert(v, e, **kw)
        with self.assertRaises(ValueError):
            self.convert([[0, 0, 0], [1, 0, 0]], [[0, 1]], size=4.5)

    def test_malformed_archive(self):
        grid = self.convert([[0, 0, 0], [1, 0, 0]], [[0, 1]])
        grid.edges[0, 0] = 10000
        with self.assertRaises(ValueError):
            flexible_dual_grid_to_lines(grid)

    def test_random_graph_topology(self):
        rng = np.random.default_rng(42)
        for size in (1, 2, 5):
            for _ in range(10):
                vertices = rng.uniform(.01, .99, (12, 3))
                edges = np.array([(i, j) for i in range(12) for j in range(i) if rng.random() < .18])
                self.assert_topology(vertices, edges, size)

    def test_cpu_import_does_not_import_gpu_dependencies(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; from o_voxel.convert import lines_to_flexible_dual_grid; "
            "assert not any(x in sys.modules for x in ('torch', 'cumesh', 'flex_gemm', 'OCP'))"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class HelmetTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("OCP"), "optional OpenCascade not installed")
    def test_step_boundaries_roundtrip(self):
        from o_voxel.convert.cad_lines import load_step_lines
        path = Path(__file__).resolve().parents[1] / "assets/helmet/24380_528e79de.step"
        v, e, ids, meta = load_step_lines(path, deflection=.1)
        self.assertEqual(meta["unique_cad_edges"], 51)
        self.assertEqual(meta["curve_types"]["GeomAbs_BSplineCurve"], 50)
        self.assertEqual(len(np.unique(ids)), 51)
        grid = lines_to_flexible_dual_grid(v, e, grid_size=32)
        self.assertEqual(topology(v, e), topology(*flexible_dual_grid_to_lines(grid)))
        nv, ne, ni, nm = load_step_lines(path, deflection=.1, nurbs_only=True)
        self.assertEqual(nm["selected_cad_edges"], 50)
        self.assertNotIn("GeomAbs_Line", nm["curve_types"])
        self.assertEqual(topology(nv, ne), topology(*flexible_dual_grid_to_lines(
            lines_to_flexible_dual_grid(nv, ne, grid_size=16))))


if __name__ == "__main__":
    unittest.main()
