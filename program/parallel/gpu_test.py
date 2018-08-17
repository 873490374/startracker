import os

import numpy as np

from program.const import MAIN_PATH


def open_old():
    input_file_path = os.path.join(
        MAIN_PATH, './program/catalog/generated/'
                   'triangle_catalog_2018_8_20_14_57.csv')

    with open(input_file_path, 'rb') as f:
        triangles = np.genfromtxt(f, dtype=np.float64, delimiter=',', skip_header=1)
    return triangles


def open_new():
    input_file_path = os.path.join(
        MAIN_PATH, './program/catalog/generated/'
                   'triangle_catalog_full_2018_8_20_14_57.csv')
    with open(input_file_path, 'rb') as f:
        triangles = np.genfromtxt(f, dtype=np.float64, delimiter=',')
    return triangles


def check():


    old = open_old()
    new = open_new()

    missing = 0

    for o in new:
        if is_missing(old, o):
            print('missing', o)
            missing += 1

    print(missing)


def is_missing(new, o):
    for n in new:
        # print(n)
        if (
                n[0] == o[0] and n[1] == o[1] and n[2] == o[2] or
                n[0] == o[0] and n[2] == o[1] and n[1] == o[2] or
                n[1] == o[0] and n[0] == o[1] and n[2] == o[2] or
                n[1] == o[0] and n[2] == o[1] and n[0] == o[2] or
                n[2] == o[0] and n[0] == o[1] and n[1] == o[2] or
                n[2] == o[0] and n[1] == o[1] and n[0] == o[2]):
            return False
    return True


check()
