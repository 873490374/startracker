import numpy as np
import pytest

from program.planar_triangle import CatalogPlanarTriangle
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene

check = [42913, 45941, 45556, 41037]


def find_stars(input_data):
    targets = []
    filename = './tests/catalog/triangle_test_catalog.csv'
    catalog = [
        CatalogPlanarTriangle(t[0], t[1], t[2], t[3], t[4])
        for t in np.genfromtxt(filename, delimiter=',')]
    for row in input_data:
        star_identifier = StarIdentifier(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=SENSOR_VARIANCE
            ),
            sensor_variance=SENSOR_VARIANCE,
            max_magnitude=MAX_MAGNITUDE,
            camera_fov=CAMERA_FOV,
            catalog=catalog)
        x = star_identifier.identify_stars(row)
        targets.append(x)
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

    @pytest.mark.skip(reason="Too long")
    def test_(self):
        input_data, result = read_scene('./tests/scenes', 'validate')
        assert len(input_data[0]) == 14
        targets = find_stars(input_data)
        assert len(targets[0]) > 0
        print(targets[0])
        print('Score: {}'.format(validate(result, targets)))
