import os

import numpy as np
import pytest
from timeit import default_timer as timer

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.tracker.cole_ident import ColeStarIdentifier
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene, read_scene_old


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
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'one_scene')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        # assert len(targets[0]) > 0
        res = targets[0][0]
        s1_id = int(res[0][0])
        s2_id = int(res[1][0])
        s3_id = int(res[2][0])
        triangle = res[3]
        sort_result = result[0]
        assert all([triangle[0] == sort_result[s1_id],
                    triangle[1] == sort_result[s2_id],
                    triangle[2] == sort_result[s3_id]])

    def test_10_scenes_mag5(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), '10_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            try:
                res = targets[i][0]
                s1_id = int(res[0][0])
                s2_id = int(res[1][0])
                s3_id = int(res[2][0])
                triangle = res[3]
                sort_result = result[i]
                if all([triangle[0] == sort_result[s1_id],
                        triangle[1] == sort_result[s2_id],
                        triangle[2] == sort_result[s3_id]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert 10 == good
        assert 0 == bad

    def test_100_scenes_mag5(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), '100_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            try:
                res = targets[i][0]
                s1_id = int(res[0][0])
                s2_id = int(res[1][0])
                s3_id = int(res[2][0])
                triangle = res[3]
                sort_result = result[i]
                if all([triangle[0] == sort_result[s1_id],
                        triangle[1] == sort_result[s2_id],
                        triangle[2] == sort_result[s3_id]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert 99 == good
        assert 1 == bad

    def test_1000_scenes_mag5(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            try:
                res = targets[i][0]
                s1_id = int(res[0][0])
                s2_id = int(res[1][0])
                s3_id = int(res[2][0])
                triangle = res[3]
                sort_result = result[i]
                if all([triangle[0] == sort_result[s1_id],
                        triangle[1] == sort_result[s2_id],
                        triangle[2] == sort_result[s3_id]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert 999 == good
        assert 1 == bad

    def test_1000_scenes_mag4(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            try:
                res = targets[i][0]
                s1_id = int(res[0][0])
                s2_id = int(res[1][0])
                s3_id = int(res[2][0])
                triangle = res[3]
                sort_result = result[i]
                if all([triangle[0] == sort_result[s1_id],
                        triangle[1] == sort_result[s2_id],
                        triangle[2] == sort_result[s3_id]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert 998 == good
        assert 2 == bad

    def test_1000_scenes_mag556(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            try:
                res = targets[i][0]
                s1_id = int(res[0][0])
                s2_id = int(res[1][0])
                s3_id = int(res[2][0])
                triangle = res[3]
                sort_result = result[i]
                if all([triangle[0] == sort_result[s1_id],
                        triangle[1] == sort_result[s2_id],
                        triangle[2] == sort_result[s3_id]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert 262 == good
        assert 738 == bad

    @pytest.mark.skip('This scenes are xy image coordinates')
    def test_esa(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'esa')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            assert len(targets[i]) > 0
            triangle = targets[i][0]
            try:
                if all([triangle[0] in result[i],
                        triangle[1] in result[i],
                        triangle[2] in result[i]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert 998 == good
        assert 2 == bad

    def test_1000_scenes_mag4_xy(self):
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        input_data, result = read_scene_old(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_xy')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area',
            kv_m, kv_q)
        good = 0
        bad = 0
        for i in range(len(targets)):
            assert len(targets[i]) > 0
            try:
                triangle = targets[i][0][3]
                if all([triangle[0] in result[i],
                        triangle[1] in result[i],
                        triangle[2] in result[i]]):
                    good += 1
                else:
                    bad += 1
            except (AttributeError, TypeError, IndexError):
                bad += 1

        print('good: ', good)
        print('bad: ', bad)
        assert good == 998
        assert bad == 2
