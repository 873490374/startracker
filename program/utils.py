import os

import numpy as np

from program.star import StarUV


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

    input_data = read_input(os.path.join(path, '{}_input.csv'.format(fname)))
    result = read_int_csv(os.path.join(path, '{}_result.csv'.format(fname)))

    return input_data, result
