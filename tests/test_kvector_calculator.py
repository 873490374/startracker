from program.planar_triangle import ImagePlanarTriangle
from program.tracker.kvector_calculator import KVectorCalculator


class TestKVector:

    def test_k_vector(self):
        y_vector = [
            ImagePlanarTriangle(None, None, None, None, 0.7, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.9, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.51, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.123, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.62, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.562, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.746, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.32, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.86, None, None),
            ImagePlanarTriangle(None, None, None, None, 0.561, None, None),
        ]
        kvector_calc = KVectorCalculator()
        kvector, m, q = kvector_calc.make_kvector(y_vector)
        expected_kvector = [
            ImagePlanarTriangle(None, None, None, None, 0.123, None, None, 0),
            ImagePlanarTriangle(None, None, None, None, 0.32, None, None, 0),
            ImagePlanarTriangle(None, None, None, None, 0.51, None, None, 0),
            ImagePlanarTriangle(None, None, None, None, 0.561, None, None, 1),
            ImagePlanarTriangle(None, None, None, None, 0.562, None, None, 1),
            ImagePlanarTriangle(None, None, None, None, 0.62, None, None, 4),
            ImagePlanarTriangle(None, None, None, None, 0.7, None, None, 5),
            ImagePlanarTriangle(None, None, None, None, 0.746, None, None, 7),
            ImagePlanarTriangle(None, None, None, None, 0.86, None, None, 8),
            ImagePlanarTriangle(None, None, None, None, 0.9, None, None, 10),
        ]
        assert m == 0.11366666666666712
        assert q == 0.009333333333330876
        assert kvector == expected_kvector

        y_a = 0.32
        y_b = 0.51
        kvector_calc.m = m
        kvector_calc.q = q
        found = kvector_calc.find_in_kvector(y_a, y_b, kvector)

        expected_found = [
            ImagePlanarTriangle(None, None, None, None, 0.32, None, None, 0),
            ImagePlanarTriangle(None, None, None, None, 0.51, None, None, 0),
            ImagePlanarTriangle(None, None, None, None, 0.561, None, None, 1),
            ImagePlanarTriangle(None, None, None, None, 0.562, None, None, 1),
        ]
        assert found == expected_found
