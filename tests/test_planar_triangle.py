import numpy as np

from startracker.planar_triangle import PlanarTriangle


class TestPlanarTriangle:

    def test_planar_triangle(self):
        p = np.matrix([1.5, 2.4, 1.3])
        q = np.matrix([14.5, 1.6, 55.1])
        r = np.matrix([16.5, 112.4, 63.5])

        planar_triangle = PlanarTriangle()
        planar_triangle.calculate_triangle(p, q, r)

        print(planar_triangle.A)
        print(planar_triangle.A_var)
        print(planar_triangle.J)
        print(planar_triangle.J_var)
