import numpy as np

import scipy.optimize


# noinspection PyPep8Naming
class QuestCalculator:
    def __init__(self):
        self.s = 0.
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
        self.s = self.calculate_s(B)
        Z = self.calculate_Z(B)
        K = self.calculate_K(S, self.s, Z)
        self.a = self.calculate_a(self.s, Z)
        self.b = self.calculate_b(self.s, Z)
        self.c = self.calculate_c(S, Z)
        self.d = self.calculate_d(S, Z)
        lamb = self.calculate_lambda(self.func)
        q = self.calculate_q(lamb, self.s, S, Z)
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
    def calculate_K(S, s, Z):
        upper = np.append(S-s * np.identity(3), Z, axis=1)
        lower = np.array([np.append(Z.T, s)])
        matrix = np.append(upper, lower, axis=0)
        return matrix

    @staticmethod
    def calculate_a(s: float, S: np.ndarray) -> float:  # scalar
        return s**2 - np.transpose(np.array(S)).trace().item()

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

    def calculate_lambda(self, func) -> float:
        x0 = 2
        return scipy.optimize.newton(
            func, x0, args=(
                self.a, self.b, self.c, self.d, self.s),
            maxiter=500)

    @staticmethod
    def calculate_q(lamb, s, S, Z):
        # TODO calculate more accurately
        arf = lamb**2 - s**2 + np.transpose(np.array(S)).trace().item()
        bet = lamb - s
        x = np.inner(arf * np.identity(3) + bet*S+S*S, Z.T).flatten()
        miu = (lamb + s) * arf - np.linalg.det(S)
        f = np.sqrt(miu**2 + x[0]**2 + x[1]**2 + x[2]**2)
        q = np.append(np.array(miu), x)
        q = q/f

        return q
