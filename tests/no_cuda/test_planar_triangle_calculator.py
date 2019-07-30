import math

import numpy as np

from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class TestPlanarTriangle:

    def test_planar_triangle(self):
        s1 = np.array([2484, 0.45034301, 0.06237546, -0.89067417])
        s2 = np.array([2487, 0.450226, 0.06238576, -0.8907326])
        s3 = np.array([2578, 0.44888572, 0.06454528, -0.89125502])
        sensor_variance = 270e-6 / 10
        sig_x = 3

        calc = PlanarTriangleCalculator(sensor_variance=sensor_variance)
        triangle = calc.calculate_triangle(s1, s2, s3)

        expected_area = 1.3412e-07
        expected_moment = 5.1856e-14
        expected_area_var = 2.5368e-15
        expected_moment_var = 3.1147e-28

        expected_area_min = -1.6980e-08
        expected_area_max = 2.8522e-07
        expected_moment_min = -1.0889e-15
        expected_moment_max = 1.0480e-13

        area = triangle[3]
        moment = triangle[4]
        area_var = triangle[5]
        moment_var = triangle[6]

        assert np.isclose(expected_area, area, atol=1.e-15)
        assert np.isclose(expected_moment, moment, atol=1.e-20)
        assert np.isclose(expected_area_var, area_var, atol=1.e-20)
        assert np.isclose(expected_moment_var, moment_var, atol=1.e-33)

        area_min = area - sig_x * math.sqrt(area_var)
        area_max = area + sig_x * math.sqrt(area_var)
        moment_min = moment - sig_x * math.sqrt(moment_var)
        moment_max = moment + sig_x * math.sqrt(moment_var)

        assert np.isclose(expected_area_min, area_min, atol=1.e-10)
        assert np.isclose(expected_area_max, area_max, atol=1.e-10)
        assert np.isclose(expected_moment_min, moment_min, atol=1.e-17)
        assert np.isclose(expected_moment_max, moment_max, atol=1.e-15)
