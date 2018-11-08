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
    targets = []
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
        ll = star_identifier.identify_stars(row)
        # print(ll)
        # print('*'*30)
        targets.append([ll])
        # print(ll)
        # print('*'*30)
        # targets.append([ll])
        times.append(timer() - start)

    print('Average time: ', np.sum(times)/len(times))
    return targets


class TestValidate:
    def test_one_scene(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'one_scene')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 1 == in_scene_good
        assert 0 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 0 == exact_good
        assert 1 == exact_bad

    def test_10_scenes_mag5(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'), '10_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 10 == in_scene_good
        assert 0 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 1 == exact_good
        assert 9 == exact_bad

    def test_100_scenes_mag5(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'), '100_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 99 == in_scene_good
        assert 1 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 0 == exact_good
        assert 100 == exact_bad

    def test_1000_scenes_mag5(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 999 == in_scene_good
        assert 1 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 5 == exact_good
        assert 995 == exact_bad

    def test_1000_scenes_mag4(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 998 == in_scene_good
        assert 2 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 7 == exact_good
        assert 993 == exact_bad

    def test_1000_scenes_mag556(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 262 == in_scene_good
        assert 738 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 1 == exact_good
        assert 999 == exact_bad

    def test_esa(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, result = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'esa',
            focal_length, (res_x, res_y))
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 5 == in_scene_good
        assert 95 == in_scene_bad

        exact_good, exact_bad = exact_stars(targets, result)
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
        input_data, result = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_xy', focal_length, (res_x, res_y))
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)

        in_scene_good, in_scene_bad = stars_in_scene(targets, result)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert 858 == in_scene_good
        assert 142 == in_scene_bad
        exact_good, exact_bad = exact_stars(targets, result)

        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert 128 == exact_good
        assert 872 == exact_bad


def stars_in_scene(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        try:
            triangle = result[i][0][3]
            if all([triangle[0] in expected[i],
                    triangle[1] in expected[i],
                    triangle[2] in expected[i]]):
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
        try:
            res = result[i][0]
            s1_id = int(res[0][0])
            s2_id = int(res[1][0])
            s3_id = int(res[2][0])
            triangle = res[3]
            expect = expected[i]
            if all([triangle[0] == expect[s1_id],
                    triangle[1] == expect[s2_id],
                    triangle[2] == expect[s3_id]]):
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad
