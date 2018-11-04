import numpy as np
import pytest

from program.const import FOCAL_LENGTH, SENSOR_VARIANCE
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.utils import (
    convert_star_to_uv,
    convert_to_vector,
)


class TestUtils:

    def test_xy_uv_conversion(self):
        uv = [
            [0.06866535, -0.0498963, 0.9963912],
            [0.07474301, 0.01140279, 0.99713763],
            [-0.06520745, -0.06038494, 0.99604299],
            [-0.02255166, 0.02891264, 0.99932751],
            [-0.07155564, -0.03705379, 0.99674812],
            [-0.08065246, -0.01520595, 0.99662629],
        ]

        orientation = np.array([
            [0.81230405, -0.16834913, -0.55840908],
            [-0.56588527, 0.00429001, -0.82447283],
            [0.14119486, 0.9857181, -0.0917815],
        ])

        pos = [
            [312.08786619, 1747.73090906],
            [54.77276898, 1678.35409499],
            [845.4802719, 137.50184802],
            [1037.46751659, 1207.62253161],
            [170.51234912, 203.81648496],
            [552.58255956, 1847.98312942],
        ]
        original_pos = [
            [170.51234912, 203.81648496],
            [845.4802719, 137.50184802],
            [54.77276898, 1678.35409499],
            [1037.46751659, 1207.62253161],
            [312.08786619, 1747.73090906],
            [552.58255956, 1847.98312942],
        ]

        ids = [26311, 25930, 24436, 26241, 23875, 26727]

        magnitudes = [
            1.68826354, 2.26076944, 0.19990068,
            2.74317044, 2.77568215, 1.74322264,
        ]

        original_ids = [23875, 24436, 25930, 26241, 26311, 26727]
        original_deg = [
            [76.96264146, -5.08626282],
            [78.63446353, -8.20163919],
            [83.00166562, -0.2990934],
            [83.85825475, -5.90989984],
            [84.05338572, -1.20191725],
            [85.18968672, -1.94257841],
        ]

        calc_uv = [convert_star_to_uv(deg) for deg in original_deg]
        calc_uv_orient = np.dot(calc_uv, orientation.transpose())
        for i in range(len(uv)):
            assert np.isclose(calc_uv_orient[i], uv[i]).all()

        pos_uv = []
        for i in range(len(uv)):
            res_x = 1920  # pixels
            res_y = 1440  # pixels
            pp = (res_x, res_y)
            a = convert_to_vector(
                original_pos[i][1], original_pos[i][0], 1,
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
            np.array([0, calc_uv[0][0], calc_uv[0][1],
                      calc_uv[0][2]]),
            np.array([1, calc_uv[1][0], calc_uv[2][1],
                      calc_uv[1][2]]),
            np.array([2, calc_uv[2][0], calc_uv[2][1],
                      calc_uv[2][2]]))

        t3 = trian_calc.calculate_triangle(
            np.array([0, pos_uv[0][0], pos_uv[0][1],
                      pos_uv[0][2]]),
            np.array([1, pos_uv[1][0], pos_uv[2][1],
                      pos_uv[1][2]]),
            np.array([2, pos_uv[2][0], pos_uv[2][1],
                      pos_uv[2][2]]))

        assert np.isclose(t1[3], t3[3])
        assert np.isclose(t1[4], t3[4])
        assert np.isclose(t1[5], t3[5])
        assert np.isclose(t1[6], t3[6])

    def test_xy_uv_conversion2(self):

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
            [-0.0546911,   0.05428271,  0.99702672],
            [-0.01278921,  0.01558816,  0.9997967 ],
            [ 0.0319105,  -0.00558917,  0.9994751 ],
            [-0.03331454, -0.04708966,  0.99833497],
            [-0.04649769, -0.04035958,  0.99810273]]

        ids = [82363, 83081, 83153, 85258, 85267]

        orientation = np.array([
            [-0.56855987, -0.63115105,  0.5276249 ],
            [-0.81017117,  0.54085489, -0.22605015],
            [-0.14269672, -0.55598952, -0.81884876]])

        calc_uv = []
        for i in range(len(alt)):
            calc_uv.append(convert_star_to_uv((alt[i], az[i])))
        calc_uv_orient = np.dot(calc_uv, orientation.transpose())
        for i in range(len(uv)):
            assert np.isclose(calc_uv_orient[i], uv[i], atol=1.e-4).all()

        pos_uv = []
        for i in range(len(uv)):
            res_x = 1920  # pixels
            res_y = 1440  # pixels
            pp = (res_x, res_y)
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
