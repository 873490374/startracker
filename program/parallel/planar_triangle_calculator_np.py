import math
from typing import Tuple

import numpy as np

from program.const import COS_CAMERA_FOV

zero_3x3 = np.matrix(np.zeros((3, 3)))


def calculate_triangle(
        s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> Tuple[float, float]:

    l1 = s1[2] * s2[2] + s1[3] * s2[3] + s1[4] * s2[4]
    l2 = s2[2] * s3[2] + s2[3] * s3[3] + s2[4] * s3[4]
    l3 = s1[2] * s3[2] + s1[3] * s3[3] + s1[4] * s3[4]

    if (
            # s1[0] != s2[0] and
            # s1[0] != s3[0] and
            # s2[0] != s3[0] and
            # s1[1] <= MAX_MAGNITUDE and
            # s2[1] <= MAX_MAGNITUDE and
            # s3[1] <= MAX_MAGNITUDE and
            l1 >= COS_CAMERA_FOV and
            l2 >= COS_CAMERA_FOV and
            l3 >= COS_CAMERA_FOV
    ):

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

        return area, moment

    return -1., -1.
