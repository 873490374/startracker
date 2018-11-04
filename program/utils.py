import os

import numpy as np

from program.const import CAMERA_FOV, FOCAL_LENGTH


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
            for i in range(int(len(raw_data_list[j])))[::4]:
                magnitude = raw_data_list[j][i]
                uv0 = raw_data_list[j][i + 1]
                uv1 = raw_data_list[j][i + 2]
                uv2 = raw_data_list[j][i + 3]
                data_list.append(np.array([int(i/4), uv0, uv1, uv2]))
            data_lists.append(data_list)
        return data_lists

    input_data = read_input(os.path.join(path, '{}_input.csv'.format(fname)))
    result = read_int_csv(os.path.join(path, '{}_result.csv'.format(fname)))

    return input_data, result


def read_scene_old(path, fname):
    def read_int_csv(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        return [
            np.array([int(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]

    def read_input_old(filename):
        pixel_size = 1
        with open(filename, 'r') as f:
            lines = f.readlines()

        raw_data_list = [np.array([
            np.float64(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]
        data_lists = []
        for j in range(len(raw_data_list)):
            data_list = []
            for i in range(int(len(raw_data_list[j])))[::3]:
                y = raw_data_list[j][i]
                x = raw_data_list[j][i + 1]
                res_x = 1920  # pixels
                res_y = 1440  # pixels
                pp = (res_x, res_y)
                u = convert_to_vector(x, y, pixel_size, FOCAL_LENGTH*res_x, pp)
                magnitude = raw_data_list[j][i + 2]
                data_list.append(np.array([int(i / 3), u[0], u[1], u[2]]))
            data_lists.append(data_list)
        return data_lists

    input_data = read_input_old(os.path.join(path, '{}_input.csv'.format(fname)))
    result = read_int_csv(os.path.join(path, '{}_result.csv'.format(fname)))

    return input_data, result


def convert_to_vector(x, y, pixel_size, focal_length, pp):
    vector = np.array([
        pixel_size * (x - (0.5 * pp[0])),
        pixel_size * (y - (0.5 * pp[1])),
        focal_length
    ])
    u = vector.T / np.linalg.norm(vector)
    return u


def convert_star_to_uv(star_positon: (float, float)) -> np.ndarray:
    """ Convert star positions to unit vector."""
    alpha = np.deg2rad(star_positon[0])  # right ascension, altitude
    delta = np.deg2rad(star_positon[1])  # declination, azimuth
    return np.array([
        np.cos(alpha) * np.cos(delta),
        np.sin(alpha) * np.cos(delta),
        np.sin(delta)
        ], dtype='float64')
