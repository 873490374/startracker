import numpy as np

from program.star import StarUV
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier

check = [42913, 45941, 45556, 41037]


def read_scene():
    def read_int_csv(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        return [
            np.array([int(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]

    def read_input(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        raw_data_list = [np.array([
            np.float64(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]
        data_lists = []
        for j in range(len(raw_data_list)):
            data_list = []
            for i in range(int(len(raw_data_list[j])/3))[::3]:
                # print(raw_data_list[j][i], raw_data_list[j][i+1], raw_data_list[j][i+2])
                alpha = raw_data_list[j][i]
                delta = raw_data_list[j][i+1]
                magnitude = raw_data_list[j][i + 2]
                data_list.append(StarUV(
                    star_id=-1,  # None
                    magnitude=magnitude,
                    unit_vector=np.array([
                        np.cos(alpha) * np.cos(delta),
                        np.sin(alpha) * np.cos(delta),
                        np.sin(delta)
                    ], dtype='float64').T
                ))
            data_lists.append(data_list)
        return data_lists

    input_data = read_input('./tests/scenes/input.csv')
    result = read_int_csv('./tests/scenes/result.csv')

    return input_data, result


def find_stars(input_data):
    targets = []
    filename = './program/catalog/generated/triangle_catalog.csv'
    catalog = [t for t in np.genfromtxt(filename, delimiter=',')]
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

    def test_(self):
        input_data, result = read_scene()
        targets = find_stars(input_data)
        print('Score: {}'.format(validate(result, targets)))
