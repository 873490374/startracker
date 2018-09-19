import os

import numpy as np

from program.const import CAMERA_FOV
from program.star import StarUV, StarPosition


def read_scene(path, fname):
    def read_int_csv(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        return [
            np.array([int(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]

    # def read_input_old(filename):
    #     focal_length = 0.5 / np.tan(np.deg2rad(CAMERA_FOV) / 2)
    #     pixel_size = 525
    #     with open(filename, 'r') as f:
    #         lines = f.readlines()
    #
    #     raw_data_list = [np.array([
    #         np.float64(x) for x in line.strip().split(',')]) for line in
    #         lines if len(line) > 1]
    #     data_lists = []
    #     for j in range(len(raw_data_list)):
    #         data_list = []
    #         for i in range(int(len(raw_data_list[j])))[::3]:
    #             y = raw_data_list[j][i]
    #             x = raw_data_list[j][i + 1]
    #             u = calc_vector(x, y, pixel_size, focal_length)
    #             magnitude = raw_data_list[j][i + 2]
    #             data_list.append(StarUV(
    #                 star_id=-1,  # None
    #                 magnitude=magnitude,
    #                 unit_vector=u
    #             ))
    #         data_lists.append(data_list)
    #     return data_lists

    def read_input(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        raw_data_list = [np.array([
            np.float64(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]
        data_lists = []
        for j in range(len(raw_data_list)):
            data_list = []
            for i in range(int(len(raw_data_list[j])))[::4]:
                magnitude = raw_data_list[j][i]
                uv0 = raw_data_list[j][i + 1]
                uv1 = raw_data_list[j][i + 2]
                uv2 = raw_data_list[j][i + 3]
                data_list.append(np.array([int(i/4), magnitude, uv0, uv1, uv2]))
            data_lists.append(data_list)
        return data_lists

    input_data = read_input(os.path.join(path, '{}_input.csv'.format(fname)))
    result = read_int_csv(os.path.join(path, '{}_result.csv'.format(fname)))

    return input_data, result


def calc_vector(x, y, pixel_size, focal_length):
    vector = np.array([
        pixel_size * x,
        pixel_size * y,
        focal_length])
    u = vector.T / np.linalg.norm(vector)
    return u


def convert_star_to_uv(star: StarPosition) -> StarUV:
    """ Convert star positions to unit vector."""
    alpha = np.deg2rad(star.right_ascension)
    delta = np.deg2rad(star.declination)
    return StarUV(
        star_id=star.id,
        magnitude=star.magnitude,
        unit_vector=np.array([
            np.cos(alpha) * np.cos(delta),
            np.sin(alpha) * np.cos(delta),
            np.sin(delta)
        ], dtype='float64').T
    )
