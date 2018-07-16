import numpy as np

from program.tracker.star_identifier import StarIdentifier

SENSOR_VARIANCE = 1


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

        raw_data_list = [np.array([x for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]
        data_lists = []
        for j in range(len(raw_data_list)):
            data_list = []
            for i in range(int(len(raw_data_list[j])/3)):
                # print(raw_data_list[j][i], raw_data_list[j][i+1], raw_data_list[j][i+2])
                data_list.append(
                    (raw_data_list[j][i], raw_data_list[j][i+1], raw_data_list[j][i+2]))
                i += 3
            data_lists.append(data_list)
        return data_lists

    input_data = read_input('./tests/scenes/input.csv')
    result = read_int_csv('./tests/scenes/result.csv')

    return input_data, result


def find_stars(input_data):
    targets = []
    for row in input_data:
        print((row[0]))
        target = []
        StarIdentifier.identify_stars(row)
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
