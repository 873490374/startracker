import os

import numpy as np
import pytest

from program.planar_triangle import CatalogPlanarTriangle
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV, MAIN_PATH
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene


def find_stars(input_data, catalog_fname, kv_m, kv_q, max_magnitude):
    targets = []
    filename = os.path.join(
        MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
    catalog = [
        CatalogPlanarTriangle(t[0], t[1], t[2], t[3], t[4], t[5])
        for t in np.genfromtxt(filename, skip_header=0, delimiter=',')]
    del catalog[0]
    for row in input_data:
        star_identifier = StarIdentifier(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=SENSOR_VARIANCE
            ),
            kvector_calculator=KVectorCalculator(kv_m, kv_q),
            sensor_variance=SENSOR_VARIANCE,
            max_magnitude=max_magnitude,
            camera_fov=CAMERA_FOV,
            catalog=catalog)
        x = star_identifier.identify_stars(row)
        targets.append([x])
    return targets


def validate(result, targets):
    score = 0

    for y, t in zip(result, targets):
        goods = np.sum((y == t) & (y != -1))
        bads = np.sum((y != t) & (y != -1))

        trues = np.max([np.sum(t != -1), 1])

        scene_score = np.max([(goods - 2 * bads) / trues, -1])

        score += scene_score

    return score


class TestValidate:

    @pytest.mark.skip('Does not work')
    def test_1(self):
        kv_m = 0.00020383375600620604
        kv_q = 6.112353943522856e-05
        max_magnitude = 3
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'a0')
        targets = find_stars(
            input_data, 'triangle_catalog_test_small_4',
            kv_m, kv_q, max_magnitude)
        assert len(targets[0]) > 0
        triangle = targets[0][0]
        print(targets[0][0])
        assert all([triangle.s1_id in result[0],
                    triangle.s2_id in result[0],
                    triangle.s3_id in result[0]])
        # print(targets[0])
        # print('Score: {}'.format(validate(result, targets)))

    @pytest.mark.skip('Too long')
    def test_2(self):
        kv_m = 4.428503935443109e-07
        kv_q = -3.378325607016865e-07
        max_magnitude = 3
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'a0')
        targets = find_stars(
            input_data, 'triangle_catalog_test_full_3',
            kv_m, kv_q, max_magnitude)
        assert len(targets[0]) > 0
        triangle = targets[0][0]
        print(targets[0][0])
        assert all([triangle.s1_id in result[0],
                    triangle.s2_id in result[0],
                    triangle.s3_id in result[0]])
        # print(targets[0])
        # print('Score: {}'.format(validate(result, targets)))

    def test_3(self):
        kv_m = 0.015470220088694078
        kv_q = 0.003400300805871943
        max_magnitude = 4
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'a1')
        targets = find_stars(
            input_data, 'triangle_catalog_test_small_one',
            kv_m, kv_q, max_magnitude)
        assert len(targets[0]) > 0
        triangle = targets[0][0]
        # print(targets[0][0])
        assert all([triangle.s1_id in result[0],
                    triangle.s2_id in result[0],
                    triangle.s3_id in result[0]])
        # print(targets[0])
        # print('Score: {}'.format(validate(result, targets)))

    @pytest.mark.skip('Too long')
    def test_4(self):
        kv_m = 4.301928959705992e-05
        kv_q = 0.0002134423209781135
        max_magnitude = 4
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'a1')
        targets = find_stars(
            input_data, 'triangle_catalog_test_small_two',
            kv_m, kv_q, max_magnitude)
        assert len(targets[0]) > 0
        triangle = targets[0][0]
        print(targets[0][0])
        assert all([triangle.s1_id in result[0],
                    triangle.s2_id in result[0],
                    triangle.s3_id in result[0]])
        # print(targets[0])
        # print('Score: {}'.format(validate(result, targets)))
