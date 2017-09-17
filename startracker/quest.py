import numpy as np

import scipy.optimize
from scipy import optimize


class QUEST:

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
