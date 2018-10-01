import numpy as np

from program.parallel.kvector_calculator_parallel import KVectorCalculator


class TestKVector:

    def test_k_vector(self):
        y_vector = np.array([
            [0, 0, 0, 0.7, 0],
            [0, 0, 0, 0.9, 0],
            [0, 0, 0, 0.51, 0],
            [0, 0, 0, 0.123, 0],
            [0, 0, 0, 0.62, 0],
            [0, 0, 0, 0.562, 0],
            [0, 0, 0, 0.746, 0],
            [0, 0, 0, 0.32, 0],
            [0, 0, 0, 0.86, 0],
            [0, 0, 0, 0.561, 0],
        ])
        kvector_calc = KVectorCalculator()
        kvector, m, q = kvector_calc.make_kvector(y_vector)
        expected_kvector = np.array([
            [0, 0, 0, 0.123, 0, 0],
            [0, 0, 0, 0.32, 0, 0],
            [0, 0, 0, 0.51, 0, 0],
            [0, 0, 0, 0.561, 0, 1],
            [0, 0, 0, 0.562, 0, 1],
            [0, 0, 0, 0.62, 0, 4],
            [0, 0, 0, 0.7, 0, 5],
            [0, 0, 0, 0.746, 0, 7],
            [0, 0, 0, 0.86, 0, 8],
            [0, 0, 0, 0.9, 0, 10],
        ])
        assert m == 0.11366666666666712
        assert q == 0.009333333333330876
        assert (kvector == expected_kvector).all()

        y_a = 0.32
        y_b = 0.51
        kvector_calc.m = m
        kvector_calc.q = q
        rangee = kvector_calc.find_in_kvector(y_a, y_b, kvector)

        expected_found = np.array([
            [0, 0, 0, 0.32, 0, 0],
            [0, 0, 0, 0.51, 0, 0],
            [0, 0, 0, 0.561, 0, 1],
            [0, 0, 0, 0.562, 0, 1],
        ])
        assert rangee == (1, 4)

        assert (kvector[1:5, :] == expected_found).all()
