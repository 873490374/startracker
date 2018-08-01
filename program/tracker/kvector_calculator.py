import math

EPSILON = 2.22 * 10 ** (-16)


class DBVector:
    def __init__(
            self, k_vector: [float], s_vector: [float], m: float, q: float):
        self.k_vector = k_vector
        self.s_vector = s_vector
        self.m = m
        self.q = q


class KVectorCalculator:

    def make_kvector(self, y_vector: [float]) -> DBVector:
        n = len(y_vector)
        y_vector = y_vector
        s_vector = sorted(y_vector)

        y_max = s_vector[n - 1]
        y_min = s_vector[0]

        delta_epsilon = (n - 1) * EPSILON

        m = (y_max + y_min + 2 * delta_epsilon) / (n - 1)
        q = y_min - m - delta_epsilon

        # y0 = y_min - self.delta_epsilon
        # yn = y_max + self.delta_epsilon

        k_vector = self.calculate_k_vector(s_vector, m, q, n)
        return DBVector(k_vector, s_vector, m, q)

    def calculate_k_vector(
            self, s_vector: [float], m: float, q: float, n: int) -> [float]:
        k_vector = [0]
        for i in range(1, len(s_vector) - 1):
            j = i
            while s_vector[j] > self.z(i, m, q):
                j -= 1
                if j == 0:
                    break
            k_vector.append(j)
        k_vector.append(n)
        return k_vector

    def z(self, x: float, m: float, q: float) -> float:
        z = m * x + q
        return z

    def find_in_kvector(
            self, y_a: float, y_b: float, db_vector: DBVector) -> [float]:
        k_vector = db_vector.k_vector
        s_vector = db_vector.s_vector
        q = db_vector.q
        m = db_vector.m

        j_b = self.calculate_j_b(y_a, q, m)
        j_t = self.calculate_j_t(y_b, q, m)

        k_start = k_vector[j_b] + 1
        k_end = k_vector[j_t]
        answer = []
        i = k_start
        while i <= k_end:
            answer.append(s_vector[i])
            i += 1
        return answer

    def calculate_j_b(self, y_a: float, q: float, m: float) -> int:
        j_b = math.floor((y_a - q) / m)
        return j_b

    def calculate_j_t(self, y_b: float, q: float, m: float) -> int:
        j_t = math.ceil((y_b - q) / m)
        return j_t
