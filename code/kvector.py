import math


class KVector:

    def __init__(self):
        self.y_vector = []
        self.s_vector = []
        self.k_vector = []
        self.n = 0                  # number of elements
        self.delta_epsilon = 0
        self.j_b = 0
        self.j_t = 0

    def make_kvector(self, y_vector):  # n, y_max, y_min):
        self.n = len(y_vector)
        self.y_vector = y_vector
        self.s_vector = sorted(self.y_vector)

        y_max = self.s_vector[self.n - 1]
        y_min = self.s_vector[0]

        epsilon = 2.22 * 10 ** (-16)
        self.delta_epsilon = (self.n - 1) * epsilon

        self.m = (y_max + y_min + 2 * self.delta_epsilon) / (self.n - 1)
        self.q = y_min - self.m - self.delta_epsilon

        y0 = y_min - self.delta_epsilon
        yn = y_max + self.delta_epsilon

        self.calculate_k_vector()

    """
    >>> sorted(student_objects, key=lambda student: student.age)  # sort by age
    [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
    """

    def calculate_k_vector(self):
        self.k_vector.append(0)
        for i in range(1, len(self.s_vector) - 1):
            j = i
            while self.s_vector[j] > self.z(i):
                j -= 1
                if j == 0:
                    break
            self.k_vector.append(j)
        self.k_vector.append(self.n)

    def z(self, x):
        z = self.m * x + self.q
        return z

    def find_in_kvector(self, y_a, y_b):

        j_b = self.calculate_j_b(y_a)
        j_t = self.calculate_j_t(y_b)

        k_start = self.k_vector[j_b] + 1
        k_end = self.k_vector[j_t]
        answer = []
        i = k_start
        while i <= k_end:
            answer.append(self.s_vector[i])
            i += 1
        return answer

    def calculate_j_b(self, y_a):
        j_b = math.floor((y_a - self.q) / self.m)
        return j_b

    def calculate_j_t(self, y_b):
        j_t = math.ceil((y_b - self.q) / self.m)
        return j_t
