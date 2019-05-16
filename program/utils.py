import os

import numpy as np


def read_scene_uv(path, fname):
    def read_int_csv(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        return [
            np.array([int(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]

    # noinspection PyUnusedLocal
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


def read_scene_xy(path, fname, focal_length, resolution):
    def read_int_csv(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        return [
            np.array([int(x) for x in line.strip().split(',')]) for line in
            lines if len(line) > 1]

    # noinspection PyUnusedLocal
    def read_input_old(filename, focal_length_, resolution_):
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
                res_x = resolution_[0]
                res_y = resolution_[1]
                pp = (0.5 * res_x, 0.5 * res_y)
                u = convert_to_vector(
                    x, y, pixel_size, focal_length_ * res_x, pp)
                magnitude = raw_data_list[j][i + 2]
                data_list.append(np.array([int(i / 3), u[0], u[1], u[2]]))
            data_lists.append(data_list)
        return data_lists

    input_data = read_input_old(
        os.path.join(path, '{}_input.csv'.format(fname)),
        focal_length, resolution)
    result = read_int_csv(os.path.join(path, '{}_result.csv'.format(fname)))

    return input_data, result


def convert_to_vector(x, y, pixel_size, focal_length, pp):
    vector = np.array([
        pixel_size * (x - pp[0]),
        pixel_size * (y - pp[1]),
        focal_length
    ])
    u = vector.T / np.linalg.norm(vector)
    return u


def convert_star_to_uv(azimuth: float, altitude: float) -> np.ndarray:
    """ Convert star positions to unit vector."""
    caz = np.cos(azimuth)
    saz = np.sin(azimuth)

    cal = np.cos(altitude)
    sal = np.sin(altitude)

    x = caz * cal
    y = saz * cal
    z = sal

    return np.array([x, y, z]).transpose()


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]


def two_common_stars_triangles(tri, tc):
    s1_id = tri[0]
    s2_id = tri[1]
    s3_id = tri[2]

    return tc[
        ((tc[:, 0] == s1_id) & (tc[:, 1] == s2_id)) |
        ((tc[:, 0] == s1_id) & (tc[:, 2] == s2_id)) |

        ((tc[:, 0] == s1_id) & (tc[:, 1] == s3_id)) |
        ((tc[:, 0] == s1_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 1] == s1_id) & (tc[:, 0] == s2_id)) |
        ((tc[:, 1] == s1_id) & (tc[:, 2] == s2_id)) |

        ((tc[:, 1] == s1_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 1] == s1_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 2] == s1_id) & (tc[:, 0] == s2_id)) |
        ((tc[:, 2] == s1_id) & (tc[:, 1] == s2_id)) |

        ((tc[:, 2] == s1_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 2] == s1_id) & (tc[:, 1] == s3_id)) |

        ((tc[:, 0] == s2_id) & (tc[:, 1] == s3_id)) |
        ((tc[:, 0] == s2_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 1] == s2_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 1] == s2_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 2] == s2_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 2] == s2_id) & (tc[:, 1] == s3_id))
    ]
