import numpy as np

import scipy.optimize


class QUEST:

    def calculate_quest(self, weight_list, w_list, v_list):
        B = self.calculate_B(weight_list, w_list, v_list)
        S = self.calculate_S(B)
        s = self.calculate_s(B)
        Z = self.calculate_Z(B)
        a = self.calculate_a(s, S)
        b = self.calculate_b(s, S)
        c = self.calculate_c(s, S)
        d = self.calculate_d(s, S)
        attitude = self.calculate_newton_raphson(quest.func)

    def calculate_B(self, weight_list, w_list, v_list):
        B_list = []
        for i in range(len(w_list)):
            B_list.append(weight_list[i]*w_list[i]*v_list[i].T)
        return B_list

    def calculate_S(self, B):
        return B + B.T

    def calculate_s(self, B):
        return B.trace()

    def calculate_Z(self, B):
        return np.ndarray([[
            B[2][3] - B[3][2],
            B[3][1] - B[1][3],
            B[1][2] - B[2][1],
        ]])

    def calculate_a(self, s, S):
        return s**2 - S.H.trace

    def calculate_b(self, s, Z):
        return s**2 + Z.T*Z

    def calculate_c(self, S, Z):
        return S.det + Z.T*S*Z

    def calculate_d(self, S, Z):
        return Z.T*S*Z

    def func(self, x):
        return (x**2 - 2)*(x**2 - 2) - 2*x + (2*2 - 3)
        # return (x**2 - a)*(x**2 - b) - c*x + (c*s - d)

    def calculate_newton_raphson(self, func):
        return scipy.optimize.newton(func, 2, maxiter=500)


quest = QUEST()
print(quest.calculate_newton_raphson(quest.func))
