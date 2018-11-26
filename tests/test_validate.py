import os

import numpy as np
from timeit import default_timer as timer

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.tracker.cole_ident import ColeStarIdentifier
from program.tracker.kvector_calculator import KVectorCalculator
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
        assert 6  # 10 == in_scene_good
        assert 4  # 0 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 6  # 10 == in_triangle_good
        assert 4  # 0 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 6  # 10 == exact_good
        assert 4  # 0 == exact_bad

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
        assert 89  # 100 == in_scene_good
        assert 11  # 0 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 89  # 100 == in_triangle_good
        assert 11  # 0 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 86  # 100 == exact_good
        assert 14  # 0 == exact_bad

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
        assert 866  # 995 == in_scene_good
        assert 134  # 5 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 866  # 994 == in_triangle_good
        assert 134  # 6 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 775  # 991 == exact_good
        assert 225  # 9 == exact_bad

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
        assert 925  # 996 == in_scene_good
        assert 75  # 4 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 925  # 996 == in_triangle_good
        assert 75  # 4 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 803  # 994 == exact_good
        assert 197  # 6 == exact_bad

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
        assert 5  # 149 == in_scene_good
        assert 995  # 851 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 5  # 147 == in_triangle_good
        assert 995  # 853 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 5  # 133 == exact_good
        assert 995  # 867 == exact_bad

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
            input_data, 'triangle_catalog_mag6_fov10_full_area',
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
        assert 630  # 853 == in_scene_good
        assert 370  # 147 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 630  # 797 == in_triangle_good
        assert 370  # 203 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 257  # 737 == exact_good
        assert 743  # 263 == exact_bad

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
        assert 548  # 881 == in_scene_good
        assert 452  # 119 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 548  # 880 == in_triangle_good
        assert 452  # 120 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 483  # 854 == exact_good
        assert 517  # 146 == exact_bad

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
        assert 686  # 887 == in_scene_good
        assert 314  # 113 == in_scene_bad

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert 686  # 864 == in_triangle_good
        assert 314  # 136 == in_triangle_bad

        exact_good, exact_bad = exact_stars(result, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 295  # 828 == exact_good
        assert 705  # 172 == exact_bad


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
                    s3 in expected_triangle]):
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
