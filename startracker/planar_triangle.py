import math
import numpy as np


zero_3x3 = np.matrix([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
])


class PlanarTriangle:

    def __init__(self):
        self.A = None
        self.J = None
        self.A_var = None
        self.J_var = None

    def calculate_triangle(self, p, q, r):
        a = np.linalg.norm(p - q)
        b = np.linalg.norm(q - r)
        c = np.linalg.norm(p - r)

        s = self.calculate_perimeter_half(a, b, c)
        self.A = self.calulate_area(s, a, b, c)
        self.J = self.calculate_polar_moment(a, b, c)

        partials = self.calculate_partial_derivatives(s, a, b, c, p, q, r)
        H = self.calculate_area_derivatives(partials)
        R = self.calculate_r_matrix(p, q, r)

        self.A_var = self.calculate_area_variance(H, R)
        self.J_var = self.calculate_polar_moment_variance(a, b, c, partials, H, R)

    def calculate_perimeter_half(self, a, b, c):
        return 0.5 * (a + b + c)

    def calulate_area(self, s, a, b, c):
        return math.sqrt(s * (s - a) * (s - b) * (s - c))

    def calculate_polar_moment(self, a, b, c):
        return self.A * (a ** 2 + b ** 2 + c ** 2) / 36

    def calculate_partial_derivatives(self, s, a, b, c, p, q, r):

        u1 = (s - a) * (s - b) * (s - c)
        u2 = s * (s - b) * (s - c)
        u3 = s * (s - a) * (s - c)
        u4 = s * (s - a) * (s - b)

        dA_da = (u1 - u2 + u3 + u4) / 4 * self.A
        dA_db = (u1 + u2 - u3 + u4) / 4 * self.A
        dA_dc = (u1 + u2 + u3 - u4) / 4 * self.A

        da_db1 = (p - q).T / a
        db_db2 = (q - r).T / b
        dc_db1 = (p - r).T / c
        da_db2 = - da_db1
        db_db3 = - db_db2
        dc_db3 = - dc_db1

        return {
            'dA_da': dA_da,
            'dA_db': dA_db,
            'dA_dc': dA_dc,
            'da_db1': da_db1,
            'db_db2': db_db2,
            'dc_db1': dc_db1,
            'da_db2': da_db2,
            'db_db3': db_db3,
            'dc_db3': dc_db3,
        }

    def calculate_area_derivatives(self, p):
        h1T = p['dA_da'] * p['da_db1'] + p['dA_dc'] * p['dc_db1']
        h2T = p['dA_da'] * p['da_db2'] + p['dA_db'] * p['db_db2']
        h3T = p['dA_db'] * p['db_db3'] + p['dA_dc'] * p['dc_db3']

        return np.array([h1T, h2T, h3T])

    def calculate_r_matrix(self, p, q, r):
        R1 = np.var(np.identity(3) - p*p.T)
        R2 = np.var(np.identity(3) - q*q.T)
        R3 = np.var(np.identity(3) - r*r.T)

        return np.matrix([
            [R1, zero_3x3, zero_3x3],
            [zero_3x3, R2, zero_3x3],
            [zero_3x3, zero_3x3, R3]
        ])

    def calculate_area_variance(self, H, R):
        # Variance - Area

        return H * R * H.T

    def calculate_polar_moment_variance(self, a, b, c, part, der, R):
        # Variance - Polar Moment

        dJ_da = self.A * a / 18
        dJ_db = self.A * b / 18
        dJ_dc = self.A * c / 18
        dJ_dA = (a**2 + b**2 + c**2) / 36

        h1T = dJ_da * part['da_db1'] + dJ_dc * part['dc_db1'] + dJ_dA * der[0]
        h2T = dJ_da * part['da_db2'] + dJ_db * part['db_db2'] + dJ_dA * der[1]
        h3T = dJ_db * part['db_db3'] + dJ_dc * part['dc_db3'] + dJ_dA * der[2]

        H = np.array([h1T, h2T, h3T])

        return H * R * H.T
