import os

import numpy as np
import pytest
from timeit import default_timer as timer

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene


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
        targets.append([star_identifier.identify_stars(row)])
        times.append(timer() - start)

    print('Average time: ', np.sum(times)/len(times))
    return targets


class TestValidate:
    @pytest.mark.skip()
    def test_one_scene(self):
        kv_m = 6.926772802907601e-10
        kv_q = -7.018966515971442e-10
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'one_scene')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full',
            kv_m, kv_q)
        assert len(targets[0]) > 0
        triangle = targets[0][0]
        assert all([triangle[0] in result[0],
                    triangle[1] in result[0],
                    triangle[2] in result[0]])

    def test_10_scenes_mag5(self):
        kv_m = 6.926772802907601e-10
        kv_q = -7.018966515971442e-10
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), '10_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full',
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
        assert good == 10
        assert bad == 0

    @pytest.mark.skip('Very, very long')
    def test_100_scenes_mag5(self):
        kv_m = 6.926772802907601e-10
        kv_q = -7.018966515971442e-10
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), '100_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full',
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
        assert good == 100
        assert bad == 0

    @pytest.mark.skip('Very, very long')
    def test_1000_scenes_mag5(self):
        kv_m = 6.926772802907601e-10
        kv_q = -7.018966515971442e-10
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), '1000_scenes_mag_5_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag4_fov10_full',
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
        assert good == 1000
        assert bad == 0

    @pytest.mark.skip('Very, very long')
    def test_1000_scenes_mag4(self):
        kv_m = 1.6302444115811607e-08
        kv_q = -1.625215694310689e-08
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), '1000_scenes_mag_4_fov_10')
        targets = find_stars(
            input_data, 'triangle_catalog_mag4_fov10_full',
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
        assert good == 1000
        assert bad == 0
