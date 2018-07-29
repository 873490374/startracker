import numpy as np

from program.star import StarUV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class TestPlanarTriangle:

    def test_planar_triangle(self):
        p = StarUV(None, None, np.array([1.5, 2.4, 1.3], dtype=np.float64))
        q = StarUV(None, None, np.array([14.5, 1.6, 55.1], dtype=np.float64))
        r = StarUV(None, None, np.array([16.5, 112.4, 63.5], dtype=np.float64))
        sensor_variance = 1

        calc = PlanarTriangleCalculator(sensor_variance=sensor_variance)
        triangle = calc.calculate_triangle(p, q, r)

        assert triangle.area == 3069.752676421994
        assert triangle.area_var == -1.7170205379962022e+21
        assert triangle.moment == 2695338.3533151103
        assert triangle.moment_var == -7.354919014967571e+27
