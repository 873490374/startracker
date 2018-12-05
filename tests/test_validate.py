import os

import numpy as np
from timeit import default_timer as timer

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.parallel.kvector_calculator_parallel import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene_uv, read_scene_xy


def find_stars(input_data, catalog_fname, kv_m, kv_q):
    results = []
    filename = os.path.join(
        MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
    with open(filename, 'rb') as f:
        catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
    times = []
    star_identifier = StarIdentifier(
        planar_triangle_calculator=PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        ),
        kvector_calculator=KVectorCalculator(kv_m, kv_q),
        catalog=catalog)
    for row in input_data:
        start = timer()
        result = star_identifier.identify_stars(row)
        results.append([result])
        times.append(timer() - start)

    print('Average time: ', np.sum(times)/len(times))
    return results


class TestValidate:
    def test_one_scene_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'), '1_scene_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 1 == in_scene_good
        assert 0 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 1 == in_triangle_good
        assert 0 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 1 == exact_good
        assert 0 == exact_bad

    def test_10_scenes_mag5_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '10_scenes_mag_5_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 10 == in_scene_good
        assert 0 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 10 == in_triangle_good
        assert 0 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 10 == exact_good
        assert 0 == exact_bad

    def test_100_scenes_mag5_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '100_scenes_mag_5_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 96 == in_scene_good
        assert 4 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 95 == in_triangle_good
        assert 5 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 95 == exact_good
        assert 5 == exact_bad

    def test_1000_scenes_mag5_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 987 == in_scene_good
        assert 13 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 976 == in_triangle_good
        assert 24 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 976 == exact_good
        assert 24 == exact_bad

    def test_1000_scenes_mag4_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 982 == in_scene_good
        assert 18 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 981 == in_triangle_good
        assert 19 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 981 == exact_good
        assert 19 == exact_bad

    def test_1000_scenes_mag556_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 9 == in_scene_good
        assert 991 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 4 == in_triangle_good
        assert 996 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 4 == exact_good
        assert 996 == exact_bad

        """
        With mag6 catalog: (total: 46 min; average: 2.7516 s)
        in scene good:  907
        in scene bad:  93
        in triangle good:  903
        in triangle bad:  97
        exact good:  903
        exact bad:  97
        """

    def test_esa_xy(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'esa_xy',
            focal_length, (res_x, res_y))
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 1 == in_scene_good
        assert 99 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 1 == in_triangle_good
        assert 99 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 1 == exact_good
        assert 99 == exact_bad

        """
        With mag6 catalog: (total: 10 min; average: 6.043 s)
        in scene good:  9
        in scene bad:  91
        in triangle good:  8
        in triangle bad:  92
        exact good:  8
        exact bad:  92
        """

    def test_1000_scenes_mag4_xy(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_xy_scramble', focal_length, (res_x, res_y))
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 843 == in_scene_good
        assert 157 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 793 == in_triangle_good
        assert 207 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 787 == exact_good
        assert 213 == exact_bad

    def test_1000_scenes_mag5_xy_no_scramble(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_xy_no_scramble',
            focal_length, (res_x, res_y))
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 864 == in_scene_good
        assert 136 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 822 == in_triangle_good
        assert 178 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 811 == exact_good
        assert 189 == exact_bad

    def test_1000_scenes_mag5_xy(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_xy_scramble',
            focal_length, (res_x, res_y))
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 904 == in_scene_good
        assert 96 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 866 == in_triangle_good
        assert 134 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 860 == exact_good
        assert 140 == exact_bad


def stars_in_scene(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        try:
            scene_result = result[i][0]
            b = 0
            for s in scene_result:
                if s[1] not in expected[i]:
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad


def stars_in_triangle(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        try:
            res = result[i][0]
            b = 0
            for s in range(len(res)-2):  # res[:-2]:
                s1 = int(res[s][1])
                s2 = int(res[s+1][1])
                s3 = int(res[s+2][1])
                expected_triangle = [
                    expected[i][s],
                    expected[i][s+1],
                    expected[i][s+2],
                ]
                if not all([
                    s1 in expected_triangle,
                    s2 in expected_triangle,
                    s3 in expected_triangle
                ]):
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad


def exact_stars(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        try:
            scene_result = result[i][0]
            b = 0
            for s in range(len(scene_result)):
                if scene_result[s][1] != expected[i][s]:
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad
