import math
import numpy as np

from program.planar_triangle import PlanarTriangleImage
from program.star import StarUV

zero_3x3 = np.matrix(np.zeros((3, 3)))


class PlanarTriangleCalculator:

    def calculate_triangle(
            self, s1: StarUV, s2: StarUV, s3: StarUV, sensor_variance: float):
        p = s1.unit_vector
        q = s2.unit_vector
        r = s3.unit_vector
        a = np.linalg.norm(p - q)
        b = np.linalg.norm(q - r)
        c = np.linalg.norm(p - r)

        s = self.calculate_perimeter_half(a, b, c)
        A = self.calulate_area(s, a, b, c)
        J = self.calculate_polar_moment(a, b, c, A)

        partials = self.calculate_partial_derivatives(s, a, b, c, p, q, r, A)
        H = self.calculate_area_derivatives(partials)
        R = self.calculate_r_matrix(p, q, r, sensor_variance)

        A_var = self.calculate_area_variance(H, R)
        J_var = self.calculate_polar_moment_variance(
            a, b, c, partials, H, R, A)

        return PlanarTriangleImage(A, J, A_var, J_var)

    def calculate_perimeter_half(self, a, b, c):
        return 0.5 * (a + b + c)

    def calulate_area(self, s, a, b, c):
        return math.sqrt(s * (s - a) * (s - b) * (s - c))

    def calculate_polar_moment(self, a, b, c, A):
        return A * (a ** 2 + b ** 2 + c ** 2) / 36

    def calculate_partial_derivatives(self, s, a, b, c, p, q, r, A):

        u1 = (s - a) * (s - b) * (s - c)
        u2 = s * (s - b) * (s - c)
        u3 = s * (s - a) * (s - c)
        u4 = s * (s - a) * (s - b)

        dA_da = (u1 - u2 + u3 + u4) / 4 * A
        dA_db = (u1 + u2 - u3 + u4) / 4 * A
        dA_dc = (u1 + u2 + u3 - u4) / 4 * A

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
        h1T = np.array(p['dA_da'] * p['da_db1'] + p['dA_dc'] * p['dc_db1']).T
        h2T = np.array(p['dA_da'] * p['da_db2'] + p['dA_db'] * p['db_db2']).T
        h3T = np.array(p['dA_db'] * p['db_db3'] + p['dA_dc'] * p['dc_db3']).T
        H = np.append(h1T, [h2T, h3T])  # H [1x9]
        return H

    def calculate_r_matrix(self, p, q, r, sensor_variance):
        R1 = sensor_variance*(np.identity(3) - np.outer(p, p))
        R2 = sensor_variance*(np.identity(3) - np.outer(q, q))
        R3 = sensor_variance*(np.identity(3) - np.outer(r, r))

        return self.build_r_matrix(R1, R2, R3)

    def build_r_matrix(self, R1, R2, R3):
        row_1 = np.concatenate((R1, zero_3x3, zero_3x3), axis=1)
        row_2 = np.concatenate((zero_3x3, R2, zero_3x3), axis=1)
        row_3 = np.concatenate((zero_3x3, zero_3x3, R3), axis=1)

        return np.concatenate((row_1, row_2, row_3), axis=0)  # R [9x9]

    def calculate_area_variance(self, H, R):
        # Variance - Area
        htr = np.array(H)[np.newaxis].T
        return (H * R * htr).item()  # scalar

    def calculate_polar_moment_variance(self, a, b, c, part, der, R, A):
        # Variance - Polar Moment

        dJ_da = A * a / 18
        dJ_db = A * b / 18
        dJ_dc = A * c / 18
        dJ_dA = (a**2 + b**2 + c**2) / 36

        h1T = np.array(
            dJ_da * part['da_db1'] + dJ_dc * part['dc_db1'] + dJ_dA * der[0])
        h2T = np.array(
            dJ_da * part['da_db2'] + dJ_db * part['db_db2'] + dJ_dA * der[1])
        h3T = np.array(
            dJ_db * part['db_db3'] + dJ_dc * part['dc_db3'] + dJ_dA * der[2])

        H = np.append(h1T, [h2T, h3T])
        htr = np.array(H)[np.newaxis].T
        return (H * R * htr).item()
