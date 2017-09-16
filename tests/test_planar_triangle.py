import numpy as np

from startracker.planar_triangle import PlanarTriangle


class TestPlanarTriangle:

    def test_planar_triangle(self):
        p = np.matrix([0, 0])
        q = np.matrix([1, 0])
        r = np.matrix([0, 1])

        planar_triangle = PlanarTriangle()
        planar_triangle.calculate_triangle(p, q, r)
