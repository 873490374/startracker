import numpy as np
from astropy.coordinates import Angle

from program.const import FOCAL_LENGTH_NORM, SENSOR_VARIANCE
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.utils import (
    convert_star_to_uv,
    convert_to_vector,
    two_common_stars_triangles,
    vector_to_angles,
)


# noinspection PyUnusedLocal
class TestUtils:

    def test_xy_uv_conversion(self):

        az = [252.44629764, 254.65512817, 254.89602773,
              261.32498828, 261.34858274]

        alt = [-59.04131648, -55.99005508, -53.16049005,
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

        attitude = np.array([
            [-0.56855987, -0.63115105,  0.52762490],
            [-0.81017117,  0.54085489, -0.22605015],
            [-0.14269672, -0.55598952, -0.81884876]])

        calc_uv = []
        for i in range(len(alt)):
            calc_uv.append(
                convert_star_to_uv(np.deg2rad(az[i]), np.deg2rad(alt[i])))
        calc_uv_orient = np.dot(calc_uv, attitude.transpose())
        for i in range(len(uv)):
            assert np.isclose(calc_uv_orient[i], uv[i], atol=1.e-4).all()

        pos_uv = []
        for i in range(len(uv)):
            res_x = 1920  # pixels
            res_y = 1440  # pixels
            pp = (0.5 * res_x, 0.5 * res_y)
            a = convert_to_vector(
                pos_orient[i][1], pos_orient[i][0], 1,
                FOCAL_LENGTH_NORM * res_x, pp)
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

    def test_uv_degrees_conversion(self):

        expected_attitude = [
            {
                'RA': Angle('18 30 43.57 hours').degree,
                'DEC': Angle('-47:53:46.3 degrees').degree,
                'AZ': Angle('181:52:53.4 degrees').degree,
                'ALT': Angle('-9:24:01.9 degrees').degree,
            },
            {
                'RA': Angle('18 29 27.97 hours').degree,
                'DEC': Angle('-47:50:45.2 degrees').degree,
                'AZ': Angle('182:05:50.0 degrees').degree,
                'ALT': Angle('-9:21:25.4 degrees').degree,
            },
            {
                'RA': Angle('18 28 08.29 hours').degree,
                'DEC': Angle('-47:34:49.1 degrees').degree,
                'AZ': Angle('182:19:58.9 degrees').degree,
                'ALT': Angle('-9:05:58.8 degrees').degree,
            },
            {
                'RA': Angle('18 26 06.23 hours').degree,
                'DEC': Angle('-47:19:16.8 degrees').degree,
                'AZ': Angle('182:41:12.4 degrees').degree,
                'ALT': Angle('-8:51:17.3 degrees').degree,
            },
            {
                'RA': Angle('18 24 47.64 hours').degree,
                'DEC': Angle('-47:06:57.6 degrees').degree,
                'AZ': Angle('182:55:32.3 degrees').degree,
                'ALT': Angle('-8:39:34.7 degrees').degree,
            },
            {
                'RA': Angle('18 24 15.05 hours').degree,
                'DEC': Angle('-46:56:35.4 degrees').degree,
                'AZ': Angle('183:01:38.8 degrees').degree,
                'ALT': Angle('-8:29:29.1 degrees').degree,
            },
            {
                'RA': Angle('19 41 57.72 hours').degree,
                'DEC': Angle('20:43:22.2 degrees').degree,
                'AZ': Angle('153:28:43.7 degrees').degree,
                'ALT': Angle('57:04:48.4 degrees').degree,
            },
            {
                'RA': Angle('6 05 02.30 hours').degree,
                'DEC': Angle('-63:23:48.4 degrees').degree,
                'AZ': Angle('161:16:16.9 degrees').degree,
                'ALT': Angle('-77:07:22.8 degrees').degree,
            },
        ]
        for s in expected_attitude:
            uv = convert_star_to_uv(np.deg2rad(s['RA']), np.deg2rad(s['DEC']))
            s2 = vector_to_angles(uv)
            assert np.isclose(s['RA'], s2[0], atol=1.e-30).all()
            assert np.isclose(s['DEC'], s2[1], atol=1.e-30).all()


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
