import numpy as np

from program.planar_triangle import PlanarTriangleCatalog
from program.star import StarUV
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier

check = [42913, 45941, 45556, 41037]


def read_scene(path, fname):
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
            for i in range(int(len(raw_data_list[j])))[::3]:
                alpha = raw_data_list[j][i]
                delta = raw_data_list[j][i + 1]
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

    input_data = read_input('{}/{}_input.csv'.format(path, fname))
    result = read_int_csv('{}/{}_result.csv'.format(path, fname))

    return input_data, result


def find_stars(input_data):
    targets = []
    filename = './program/catalog/generated/triangle_catalog.csv'
    catalog = [
        PlanarTriangleCatalog(t[0], t[1], t[2], t[3], t[4])
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


class TestUtils:

    def test_read_scene(self):
        input_data, result = read_scene('./tests/scenes', 'scene_read_test')
        assert len(input_data) == 1
        assert len(result) == 1

        valid_input_data = np.array([
            StarUV(-1, 1.899234433685227,
                   np.array([0.34191118, 0.93017646, -0.13367307])),
            StarUV(-1, 2.1408339115513795,
                   np.array([0.49165827, 0.55138579, 0.67397764])),
            StarUV(-1, 2.9690640538571267,
                   np.array([0.85537044, -0.00486044, 0.517994])),
            StarUV(-1, 1.9298043460377514,
                   np.array([0.01084465, -0.99385139, -0.11018987])),
            StarUV(-1, 2.7608854416083433,
                   np.array([-0.32097292, 0.90513446, -0.27876152])),
            StarUV(-1, 2.569904849798762,
                   np.array([0.92571613, -0.11994349, -0.35869655])),
            StarUV(-1, 1.6483784281102398,
                   np.array([0.03385701, -0.70661934, -0.70678343])),
            StarUV(-1, 3.001700957650252,
                   np.array([0.08804479, -0.40541719, -0.90988187])),
            StarUV(-1, 2.4727375359373367,
                   np.array([-0.00482872, -0.03424387, 0.99940184])),
            StarUV(-1, 2.4813444701656047,
                   np.array([0.01163716, -0.01209152, 0.99985918])),
            StarUV(-1, 2.2082238181410747,
                   np.array([0.46999456, -0.48154559, -0.73974249])),
            StarUV(-1, 2.8808625636857403,
                   np.array(
                       [-3.04004076e-04, 1.38796274e-01, -9.90320909e-01])),
            StarUV(-1, 1.9795735855128245,
                   np.array([0.73373499, 0.43550017, -0.5215099])),
            StarUV(-1, 1.8644780398686789,
                   np.array([0.80803054, -0.43016093, 0.40255213])),
        ])
        assert np.array_equal(input_data[0], valid_input_data)
        assert np.array_equal(result[0], np.array([
            -1, -1, -1, 42913, -1, -1, -1, -1, 45941, -1, 45556, -1, -1, 41037
        ]))


class TestValidate:

    def test_(self):
        input_data, result = read_scene('./tests/scenes', 'validate')
        assert len(input_data[0]) == 14
        targets = find_stars(input_data)
        assert len(targets[0]) > 0
        print(targets[0])
        print('Score: {}'.format(validate(result, targets)))
