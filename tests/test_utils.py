import numpy as np

from program.const import FOCAL_LENGTH, SENSOR_VARIANCE
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.utils import (
    convert_star_to_uv,
    convert_to_vector,
    two_common_stars_triangles)


# noinspection PyUnusedLocal
class TestUtils:

    def test_xy_uv_conversion(self):

        alt = [252.44629764, 254.65512817, 254.89602773,
               261.32498828, 261.34858274]

        az = [-59.04131648, -55.99005508, -53.16049005,
              -55.52982397, -56.37768824]

        pos_orient = [
            [1317.41229976, 1561.90693577],
            [ 891.08134844, 1100.36265625],
            [ 658.63864341,  609.66693069],
            [ 202.43044951, 1326.16509805],
            [ 276.29853801, 1471.18199192]]

        uv = [
            [ 0.33131708,  0.03083089,  0.94301561],
            [ 0.85249444,  0.47047207, -0.22783605],
            [-0.88413344, -0.44271065, -0.14938322],
            [ 0.87918149,  0.01082032, -0.47636419],
            [ 0.90264688,  0.32272527,  0.28474025]]

        ids = [82363, 83081, 83153, 85258, 85267]

        orientation = np.array([
            [-0.56855987, -0.63115105,  0.52762490],
            [-0.81017117,  0.54085489, -0.22605015],
            [-0.14269672, -0.55598952, -0.81884876]])

        calc_uv = []
        for i in range(len(alt)):
            calc_uv.append(convert_star_to_uv(alt[i], az[i]))
        calc_uv_orient = np.dot(calc_uv, orientation.transpose())
        for i in range(len(uv)):
            assert np.isclose(calc_uv_orient[i], uv[i], atol=1.e-4).all()

        pos_uv = []
        for i in range(len(uv)):
            res_x = 1920  # pixels
            res_y = 1440  # pixels
            pp = (0.5 * res_x, 0.5 * res_y)
            a = convert_to_vector(
                pos_orient[i][1], pos_orient[i][0], 1,
                FOCAL_LENGTH * res_x, pp)
            b = uv[i]
            # FIXME why pos[0] is always inverse of uv[0]?
            assert np.isclose(np.abs(a), np.abs(b)).all()
            pos_uv.append(a)

        trian_calc = PlanarTriangleCalculator(SENSOR_VARIANCE)
        t1 = trian_calc.calculate_triangle(
            np.array([0, calc_uv_orient[0][0], calc_uv_orient[0][1],
                      calc_uv_orient[0][2]]),
            np.array([1, calc_uv_orient[1][0], calc_uv_orient[2][1],
                      calc_uv_orient[1][2]]),
            np.array([2, calc_uv_orient[2][0], calc_uv_orient[2][1],
                      calc_uv_orient[2][2]]))

        t2 = trian_calc.calculate_triangle(
            np.array([0, uv[0][0], uv[0][1],
                      uv[0][2]]),
            np.array([1, uv[1][0], uv[2][1],
                      uv[1][2]]),
            np.array([2, uv[2][0], uv[2][1],
                      uv[2][2]]))

        t3 = trian_calc.calculate_triangle(
            np.array([0, pos_uv[0][0], pos_uv[0][1],
                      pos_uv[0][2]]),
            np.array([1, pos_uv[1][0], pos_uv[2][1],
                      pos_uv[1][2]]),
            np.array([2, pos_uv[2][0], pos_uv[2][1],
                      pos_uv[2][2]]))

        assert np.isclose(t1[3], t3[3], atol=1.e-6)
        assert np.isclose(t1[4], t3[4])
        assert np.isclose(t1[5], t3[5])
        assert np.isclose(t1[6], t3[6])

        assert np.isclose(t1[3], t2[3], atol=1.e-6)
        assert np.isclose(t1[4], t2[4])
        assert np.isclose(t1[5], t2[5])
        assert np.isclose(t1[6], t2[6])

        assert np.isclose(t2[3], t3[3])
        assert np.isclose(t2[4], t3[4])
        assert np.isclose(t2[5], t3[5])
        assert np.isclose(t2[6], t3[6])


class TestStarIdentifier:

    def test_common_triangles(self):
        tri = np.array([5, 6, 7, 231, 515])
        tc = np.array([
            [1, 2, 3, 222, 232],
            [2, 4, 7, 222, 232],
            [1, 4, 7, 222, 232],
            [6, 7, 9, 222, 232],
            [3, 5, 6, 222, 232],
            [3, 6, 7, 222, 232],
        ])
        result = two_common_stars_triangles(tri, tc)
        expected = np.array([
            [6, 7, 9, 222, 232],
            [3, 5, 6, 222, 232],
            [3, 6, 7, 222, 232],
        ])
        assert (expected == result).all()
