import numpy as np

import scipy.optimize


class QuestCalculator:

    def calculate_quest(self, weight_list, w_list, v_list):
        B = self.calculate_B(weight_list, w_list, v_list)
        S = self.calculate_S(B)
        self.s = self.calculate_s(B)
        Z = self.calculate_Z(B)
        self.a = self.calculate_a(self.s, Z)
        self.b = self.calculate_b(self.s, Z)
        self.c = self.calculate_c(S, Z)
        self.d = self.calculate_d(S, Z)
        return self.calculate_newton_raphson(self.func)

    def calculate_B(self, weight_list, w_list, v_list):
        B_list = []
        B = np.zeros((3, 3))
        for i in range(len(w_list)):
            B_list.append(weight_list[i]*w_list[i]*v_list[i].T)
        for y in range(3):
            for x in range(3):
                B[x][y] = self.sum_element(x, y, B_list)
        return B

    def sum_element(self, x, y, matrices_list):
        d = []
        for l in range(len(matrices_list)):
            d.append(matrices_list[l].item(x, y))
        return sum(d)

    def calculate_S(self, B):  # 3x3
        return B + B.T

    def calculate_s(self, B):  # scalar
        return B.trace()

    def calculate_Z(self, B):  # 1x3
        return np.array([
            B.item(1, 2) - B.item(2, 1),
            B.item(2, 0) - B.item(0, 2),
            B.item(0, 1) - B.item(1, 0)
        ])[np.newaxis].T

    def calculate_a(self, s, S):  # scalar
        return (s**2 - np.matrix(S).H.trace()).item()

    def calculate_b(self, s, Z):  # scalar
        return s**2 + np.inner(Z.T, Z.T).item()

    def calculate_c(self, S, Z):  # scalar
        z_s = np.dot(Z.T, S)
        z_s_z = np.inner(z_s, Z.T).item()
        return np.linalg.det(S) + z_s_z

    def calculate_d(self, S, Z):  # scalar
        z_s = np.dot(Z.T, S)
        return np.inner(z_s, Z.T).item()

    def func(self, x, a, b, c, d, s):
        # return (x**2 - 2)*(x**2 - 2) - 2*x + (2*2 - 3)
        return (x**2 - a)*(x**2 - b) - c*x + (c*s - d)

    def calculate_newton_raphson(self, func):
        x0 = 2
        return scipy.optimize.newton(
            func, x0, args=(
                self.a, self.b, self.c, self.d, self.s),
            maxiter=500)
