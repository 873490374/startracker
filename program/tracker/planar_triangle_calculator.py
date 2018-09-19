import math

import numpy as np


zero_3x3 = np.matrix(np.zeros((3, 3)))


class PlanarTriangleCalculator:
    def __init__(self, sensor_variance: float):
        self.sensor_variance = sensor_variance

    def calculate_triangle(
            self, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
    ) -> np.ndarray:

        p = np.array([s1[2], s1[3], s1[4]])
        q = np.array([s2[2], s2[3], s2[4]])
        r = np.array([s3[2], s3[3], s3[4]])

        a1 = s1[2] - s2[2]
        a2 = s1[3] - s2[3]
        a3 = s1[4] - s2[4]
        b1 = s2[2] - s3[2]
        b2 = s2[3] - s3[3]
        b3 = s2[4] - s3[4]
        c1 = s1[2] - s3[2]
        c2 = s1[3] - s3[3]
        c3 = s1[4] - s3[4]

        a = math.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
        b = math.sqrt(b1 ** 2 + b2 ** 2 + b3 ** 2)
        c = math.sqrt(c1 ** 2 + c2 ** 2 + c3 ** 2)

        s = 0.5 * (a + b + c)
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        moment = area * (a ** 2 + b ** 2 + c ** 2) / 36

        partials = self.calculate_partial_derivatives(
            s, a, b, c, p, q, r, area)
        H = self.calculate_area_derivatives(partials)
        R = self.calculate_r_matrix(p, q, r, self.sensor_variance)

        area_var = self.calculate_area_variance(H, R)
        moment_var = self.calculate_polar_moment_variance(
            a, b, c, partials, H, R, area)

        return np.array(
            [s1[0], s2[0], s3[0], area, moment, area_var, moment_var])

    def calculate_partial_derivatives(
            self, s: float, a: float, b: float, c: float, p: np.ndarray,
            q: np.ndarray, r: np.ndarray, area: float) -> {np.ndarray}:

        u1 = (s - a) * (s - b) * (s - c)
        u2 = s * (s - b) * (s - c)
        u3 = s * (s - a) * (s - c)
        u4 = s * (s - a) * (s - b)

        dA_da = (u1 - u2 + u3 + u4) / 4 * area
        dA_db = (u1 + u2 - u3 + u4) / 4 * area
        dA_dc = (u1 + u2 + u3 - u4) / 4 * area

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

    def calculate_area_derivatives(self, p: {np.ndarray}) -> np.ndarray:
        h1T = np.array(p['dA_da'] * p['da_db1'] + p['dA_dc'] * p['dc_db1']).T
        h2T = np.array(p['dA_da'] * p['da_db2'] + p['dA_db'] * p['db_db2']).T
        h3T = np.array(p['dA_db'] * p['db_db3'] + p['dA_dc'] * p['dc_db3']).T
        H = np.append(h1T, [h2T, h3T])  # H [1x9]
        return H

    def calculate_r_matrix(
            self, p: np.ndarray, q: np.ndarray, r: np.ndarray,
            sensor_variance: float) -> np.ndarray:
        R1 = sensor_variance*(np.identity(3) - np.outer(p, p))
        R2 = sensor_variance*(np.identity(3) - np.outer(q, q))
        R3 = sensor_variance*(np.identity(3) - np.outer(r, r))

        return self.build_r_matrix(R1, R2, R3)

    def build_r_matrix(
            self, R1: np.ndarray, R2: np.ndarray,
            R3: np.ndarray) -> np.ndarray:
        row_1 = np.concatenate((R1, zero_3x3, zero_3x3), axis=1)
        row_2 = np.concatenate((zero_3x3, R2, zero_3x3), axis=1)
        row_3 = np.concatenate((zero_3x3, zero_3x3, R3), axis=1)

        return np.concatenate((row_1, row_2, row_3), axis=0)  # R [9x9]

    def calculate_area_variance(
            self, H: np.ndarray, R: np.ndarray) -> float:
        # Variance - Area
        H_transpose = np.array(H)[np.newaxis].T
        return (H * R * H_transpose).item()  # scalar

    def calculate_polar_moment_variance(
            self, a: float, b: float, c: float, part: {np.ndarray},
            der: np.ndarray, R, area: float) -> float:
        # Variance - Polar Moment

        dJ_da = area * a / 18
        dJ_db = area * b / 18
        dJ_dc = area * c / 18
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
