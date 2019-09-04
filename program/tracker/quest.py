import numpy as np

import scipy.optimize


# noinspection PyPep8Naming
class QuestCalculator:
    def __init__(self):
        self.sigma = 0.
        self.a = 0.
        self.b = 0.
        self.c = 0.
        self.d = 0.

    def calculate_quest(
            self, weight_list: [float],
            v_b_list: [float],  # body frame vector / new measured
            v_i_list: [float],  # inertial frame vector / catalog
    ) -> (np.ndarray, np.ndarray):
        B = self.calculate_B(weight_list, v_b_list, v_i_list)
        S = self.calculate_S(B)
        self.sigma = self.calculate_s(B)
        Z = self.calculate_Z(B)
        K = self.calculate_K(S, self.sigma, Z)
        delta = np.linalg.det(S)
        adjS = np.linalg.det(S) * np.linalg.inv(S)
        kappa = np.trace(adjS)
        self.a = self.calculate_a(self.sigma, kappa)
        self.b = self.calculate_b(self.sigma, Z)
        self.c = self.calculate_c(S, Z)
        self.d = self.calculate_d(S, Z)
        try:
            lamb = self.calculate_lambda(len(v_b_list), self.func)
        except RuntimeError:
            return None, K
        q = self.calculate_q(lamb, self.sigma, S, Z, kappa, delta)
        return q, K

    @staticmethod
    def calculate_B(
            weight_list: [float], v_b_list: [float], v_i_list: [float]
    ) -> np.ndarray:
        B_list = []
        B = np.zeros((3, 3))
        for i in range(len(v_i_list)):
            B_list.append(
                weight_list[i] * np.outer(v_b_list[i].T, v_i_list[i]))
        for m in range(len(B_list)):
            B = np.add(B, B_list[m])
        return B

    @staticmethod
    def calculate_S(B: np.ndarray) -> np.ndarray:  # 3x3
        return B + B.T

    @staticmethod
    def calculate_s(B: np.ndarray) -> float:  # scalar
        return B.trace()

    @staticmethod
    def calculate_Z(B: np.ndarray) -> np.ndarray:  # 1x3
        return np.array([
            B.item(1, 2) - B.item(2, 1),
            B.item(2, 0) - B.item(0, 2),
            B.item(0, 1) - B.item(1, 0)
        ])[np.newaxis].T

    @staticmethod
    def calculate_K(
            S: np.ndarray, s: float, Z: np.ndarray) -> np.ndarray:  #3x3
        upper = np.append(S-s * np.identity(3), Z, axis=1)
        lower = np.array([np.append(Z.T, s)])
        matrix = np.append(upper, lower, axis=0)
        return matrix

    @staticmethod
    def calculate_a(s: float, kappa: float) -> float:  # scalar
        return s**2 - kappa

    @staticmethod
    def calculate_b(s: float, Z: np.ndarray) -> float:  # scalar
        return s**2 + np.inner(Z.T, Z.T).item()

    @staticmethod
    def calculate_c(S: np.ndarray, Z: np.ndarray) -> float:  # scalar
        z_s = np.dot(Z.T, S)
        z_s_z = np.inner(z_s, Z.T).item()
        return np.linalg.det(S) + z_s_z

    @staticmethod
    def calculate_d(S: np.ndarray, Z: np.ndarray) -> float:  # scalar
        z_s = np.dot(Z.T, S)
        return np.inner(z_s, Z.T).item()

    @staticmethod
    def func(x: float, a: float, b: float, c: float, d: float, s: float):
        return (x**2 - a)*(x**2 - b) - c*x + (c*s - d)

    def calculate_lambda(self, x0, func) -> float:
        return scipy.optimize.newton(
            func, x0, args=(
                self.a, self.b, self.c, self.d, self.sigma),
            maxiter=100000)

    @staticmethod
    def calculate_q(
            lamb: float, sigma: float, S: np.ndarray, Z: np.ndarray,
            kappa: float, delta: float):
        alpha = lamb ** 2 - sigma ** 2 + kappa
        beta = lamb - sigma
        gamma = (lamb + sigma) * alpha - delta
        x = np.dot(alpha * np.identity(3) + beta * S + np.dot(S, S), Z)
        f = np.sqrt(gamma**2 + x[0]**2 + x[1]**2 + x[2]**2)
        q = np.append(x, gamma)
        return q/f
