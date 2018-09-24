import math

import numpy as np

from program.planar_triangle import ImagePlanarTriangle

EPSILON = 2.22 * 10 ** (-16)


class KVectorCalculator:

    def __init__(self, m: float=None, q: float=None):
        self.m = m
        self.q = q

    def make_kvector(
            self, y_vector: np.ndarray) -> (np.ndarray, float, float):
        n = len(y_vector)
        y_vector = y_vector[y_vector[:, 3].argsort()]
        s_vector = np.zeros((len(y_vector), 6))
        s_vector[:, :-1] = y_vector

        y_max = s_vector[n - 1]
        y_min = s_vector[0]

        delta_epsilon = (n - 1) * EPSILON

        m = (y_max[3] + y_min[3] + 2 * delta_epsilon) / (n - 1)
        q = y_min[3] - m - delta_epsilon

        # y0 = y_min - self.delta_epsilon
        # yn = y_max + self.delta_epsilon

        k_vector = self.calculate_k_vector(s_vector, m, q, n)
        print('m: {}; q: {}'.format(m, q))
        return k_vector, m, q

    def calculate_k_vector(
            self, s_vector: np.ndarray, m: float, q: float, n: int) -> [float]:
        # k_vector = [0]
        s_vector[0][5] = 0
        s_vector[-1][5] = n
        for i in range(1, n - 1):
            j = i
            while s_vector[j][3] > self.z(i, m, q):
                j -= 1
                if j == 0:
                    break
            s_vector[i][5] = j
        # k_vector.append(n)
        return s_vector

    def z(self, x: float, m: float, q: float) -> float:
        z = m * x + q
        return z

    def find_in_kvector(
            self, y_a: float, y_b: float, k_vector: [ImagePlanarTriangle],
            m: float=None, q: float=None) -> [float]:
        m = m or self.m
        q = q or self.q

        j_b = max(self.calculate_j_b(y_a, m, q), 0)
        j_t = min(self.calculate_j_t(y_b, m, q), len(k_vector)-1)

        # if j_b > 0:
        #     print('j_b: ', j_b)
        # if j_t < len(k_vector)-1:
        #     print('j_t: ', j_t)

        if j_b > len(k_vector) + 1 or j_t < 0:
            return []

        k_start = int(k_vector[j_b].k + 1)
        k_end = min(int(k_vector[j_t].k), len(k_vector)-1)
        # return k_vector[k_start:k_end]
        answer = []
        i = k_start
        while i <= k_end:
            answer.append(k_vector[i])
            i += 1
        return answer

    def calculate_j_b(self, y_a: float, m: float, q: float) -> int:
        j_b = math.floor((y_a - q) / m)
        return j_b

    def calculate_j_t(self, y_b: float, m: float, q: float) -> int:
        j_t = math.ceil((y_b - q) / m)
        return j_t
