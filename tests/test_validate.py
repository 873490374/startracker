import numpy as np

from program.star import StarUV
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.star_identifier import StarIdentifier


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
    for row in input_data:
        # print((row[0]))
        # print((type(row[0])))
        # with open('./program/catalog/generated/triangle_catalog.csv', 'r') as filename:
        filename = './program/catalog/generated/triangle_catalog.csv'
            # csvwriter = csv.DictWriter(csvfile, fieldnames=[
            #     'star1_id', 'star2_id', 'star3_id', 'area', 'polar_moment'])
        catalog = np.genfromtxt(filename, delimiter=',')
            #     filename, sep='|', skipinitialspace=True,
            #     names=[
            #         'star1_id', 'star2_id', 'star3_id', 'area', 'polar_moment']
            # )
        # print(catalog)
        star_identifier = StarIdentifier(
            SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV, catalog)
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
