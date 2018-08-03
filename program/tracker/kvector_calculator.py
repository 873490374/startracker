import math
import operator

from program.planar_triangle import ImagePlanarTriangle

EPSILON = 2.22 * 10 ** (-16)


class KVectorCalculator:

    def __init__(self, m: float=None, q: float=None):
        self.m = m
        self.q = q

    def make_kvector(
            self, y_vector: [ImagePlanarTriangle]
    ) -> [ImagePlanarTriangle]:
        n = len(y_vector)
        s_vector = sorted(y_vector, key=operator.attrgetter('moment'))

        y_max = s_vector[n - 1]
        y_min = s_vector[0]

        delta_epsilon = (n - 1) * EPSILON

        m = (y_max.moment + y_min.moment + 2 * delta_epsilon) / (n - 1)
        q = y_min.moment - m - delta_epsilon

        # y0 = y_min - self.delta_epsilon
        # yn = y_max + self.delta_epsilon

        k_vector = self.calculate_k_vector(s_vector, m, q, n)
        print('m: {}; q: {}'.format(m, q))
        return k_vector, m, q

    def calculate_k_vector(
            self, s_vector: [ImagePlanarTriangle],
            m: float, q: float, n: int) -> [float]:
        # k_vector = [0]
        s_vector[0].k = 0
        s_vector[-1].k = n
        for i in range(1, n - 1):
            j = i
            while s_vector[j].moment > self.z(i, m, q):
                j -= 1
                if j == 0:
                    break
            s_vector[i].k = j
        # k_vector.append(n)
        return s_vector

    def z(self, x: float, m: float, q: float) -> float:
        z = m * x + q
        return z

    def find_in_kvector(
            self, y_a: float, y_b: float, k_vector: [ImagePlanarTriangle],
            m: float=None, q: float=None) -> [float]:
        # k_vector = db_vector.k_vector
        # s_vector = db_vector.s_vector
        m = m or self.m
        q = q or self.q

        j_b = self.calculate_j_b(y_a, m, q)
        j_t = self.calculate_j_t(y_b, m, q)

        k_start = k_vector[j_b].k + 1
        k_end = k_vector[j_t].k
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
