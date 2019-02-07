import os

import numpy as np
from timeit import default_timer as timer

import pytest

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.parallel.kvector_calculator_parallel import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene_uv, read_scene_xy


def find_stars(input_data, catalog_fname, kv_m, kv_q, verify_stars_flag):
    results = []
    filename_triangle = os.path.join(
        MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
    filename_star = os.path.join(
        MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
    with open(filename_triangle, 'rb') as f:
        triangle_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
    with open(filename_star, 'rb') as f:
        star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
    times = []
    star_identifier = StarIdentifier(
        planar_triangle_calculator=PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        ),
        kvector_calculator=KVectorCalculator(kv_m, kv_q),
        triangle_catalog=triangle_catalog,
        star_catalog=star_catalog,
        verify_stars_flag=verify_stars_flag
    )
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
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 1
        assert in_scene_bad == 0
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 1
        assert in_scene_bad_verify == 0

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 1
        assert in_triangle_bad == 0
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 1
        assert in_triangle_bad_verify == 0

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 1
        assert exact_bad == 0
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 1
        assert exact_bad_verify == 0

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 1.0
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 1.0

    def test_10_scenes_mag5_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '10_scenes_mag_5_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 10
        assert in_scene_bad == 0
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 10
        assert in_scene_bad_verify == 0

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 10
        assert in_triangle_bad == 0
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 10
        assert in_triangle_bad_verify == 0

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 10
        assert exact_bad == 0
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 10
        assert exact_bad_verify == 0

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 1.0
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 1.0

    def test_100_scenes_mag5_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '100_scenes_mag_5_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 96
        assert in_scene_bad == 4
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 100
        assert in_scene_bad_verify == 0

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 95
        assert in_triangle_bad == 5
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 99
        assert in_triangle_bad_verify == 1

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 95
        assert exact_bad == 5
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 99
        assert exact_bad_verify == 1

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9769166666666667
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9752500000000001

    def test_1000_scenes_mag5_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 987
        assert in_scene_bad == 13
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 998
        assert in_scene_bad_verify == 2

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 976
        assert in_triangle_bad == 24
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 987
        assert in_triangle_bad_verify == 13

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 976
        assert exact_bad == 24
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 987
        assert exact_bad_verify == 13

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9895348484848484
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9888681818181818

    def test_1000_scenes_mag4_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 982
        assert in_scene_bad == 18
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 983
        assert in_scene_bad_verify == 17

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 981
        assert in_triangle_bad == 19
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 982
        assert in_triangle_bad_verify == 18

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 981
        assert exact_bad == 19
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 982
        assert exact_bad_verify == 18

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9969047619047622
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9969047619047622

    def test_1000_scenes_mag556_uv(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 9
        assert in_scene_bad == 991
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 917
        assert in_scene_bad_verify == 83

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 4
        assert in_triangle_bad == 996
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 894
        assert in_triangle_bad_verify == 106

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 4
        assert exact_bad == 996
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 894
        assert exact_bad_verify == 106

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.3920965239579192
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.33892573034664575

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
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 1
        assert in_scene_bad == 99
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 89
        assert in_scene_bad_verify == 11

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 1
        assert in_triangle_bad == 99
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 88
        assert in_triangle_bad_verify == 12

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 1
        assert exact_bad == 99
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 88
        assert exact_bad_verify == 12

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.08881358420561206
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.4466271059092267

    @pytest.mark.skip('Too long test')
    def test_1000_scenes_mag556_uv_cat_mag6(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10_uv')
        result = find_stars(
            input_data, 'triangle_catalog_mag6_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag6_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 907
        assert in_scene_bad == 93
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 985
        assert in_scene_bad_verify == 15

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 903
        assert in_triangle_bad == 97
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 985
        assert in_triangle_bad_verify == 15

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 903
        assert exact_bad == 97
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 985
        assert exact_bad_verify == 15

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9579092482680718
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9535647316735554

        """
        Total time: 1h 14m 34s
        Average time: 2.2561510484499596
        Average time: 2.2021919318470684
        """

    @pytest.mark.skip('Too long test')
    def test_esa_xy_cat_mag6(self):
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
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag6_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 9
        assert in_scene_bad == 91
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 63
        assert in_scene_bad_verify == 37

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 8
        assert in_triangle_bad == 92
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 62
        assert in_triangle_bad_verify == 38

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 8
        assert exact_bad == 92
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 62
        assert exact_bad_verify == 38

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.2943908145947621
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.5633633926273245

        """
        Total time: 16m 57s
        Average time: 5.081271586679832
        Average time: 4.9285306808902165
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
            '1000_scenes_mag_4_fov_10_xy_scramble',
            focal_length, (res_x, res_y))
        result = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 843
        assert in_scene_bad == 157
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 977
        assert in_scene_bad_verify == 23

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 793
        assert in_triangle_bad == 207
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 907
        assert in_triangle_bad_verify == 93

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 787
        assert exact_bad == 213
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 896
        assert exact_bad_verify == 104

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9537619047619064
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9526369047619064

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
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 864
        assert in_scene_bad == 136
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 992
        assert in_scene_bad_verify == 8

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 822
        assert in_triangle_bad == 178
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 938
        assert in_triangle_bad_verify == 62

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 811
        assert exact_bad == 189
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 921
        assert exact_bad_verify == 79

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9482375152625161
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9460946581196591

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
            kv_m, kv_q, False)
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q, True)

        in_scene_good, in_scene_bad = stars_in_scene(result, expected)
        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good: ', in_scene_good)
        print('in scene bad: ', in_scene_bad)
        assert in_scene_good == 904
        assert in_scene_bad == 96
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 994
        assert in_scene_bad_verify == 6

        in_triangle_good, in_triangle_bad = stars_in_triangle(result, expected)
        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good: ', in_triangle_good)
        print('in triangle bad: ', in_triangle_bad)
        assert in_triangle_good == 866
        assert in_triangle_bad == 134
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 946
        assert in_triangle_bad_verify == 54

        exact_good, exact_bad = exact_stars(result, expected)
        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good: ', exact_good)
        print('exact bad: ', exact_bad)
        assert exact_good == 860
        assert exact_bad == 140
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 938
        assert exact_bad_verify == 62

        percent = percent_stars_found(result, expected)
        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified: ', percent)
        assert percent == 0.9606980658230673
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 0.9598885420135435


def stars_in_scene(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        try:
            scene_result = result[i][0]
            b = 0
            for s in scene_result:
                if s[1] not in expected[i] and s[1] != -1:
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
                    -1,
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
                if (scene_result[s][1] != expected[i][s] and
                        scene_result[s][1] != -1):
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad


def percent_stars_found(result, expected):
    percent = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        p = 0
        try:
            scene_result = result[i][0]
            for s in range(len(scene_result)):
                if scene_result[s][1] == expected[i][s]:
                    p += 1
            p = p/len(scene_result)
        except (AttributeError, TypeError, IndexError):
            percent += 0
        percent += p
    percent = percent/len(result)
    return percent
