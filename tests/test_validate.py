import os

import numpy as np
import pytest

from program.planar_triangle import CatalogPlanarTriangle
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV, MAIN_PATH
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene


def find_stars(input_data, catalog_fname, kv_m, kv_q):
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
            max_magnitude=MAX_MAGNITUDE,
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

    def test_(self):
        kv_m = 0
        kv_q = 0
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'b0')
        targets = find_stars(
            input_data, 'triangle_catalog_test_small_one', kv_m, kv_q)
        assert len(targets[0]) > 0
        triangle = targets[0][0]
        print(targets[0][0])
        assert all([triangle.s1_id in result[0],
                    triangle.s2_id in result[0],
                    triangle.s3_id in result[0]])
        # print(targets[0])
        # print('Score: {}'.format(validate(result, targets)))
