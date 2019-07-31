import os

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# noinspection PyPackageRequirements
import pytest
from PIL import Image

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.utils import read_scene_uv, read_scene_xy


def find_stars(input_data, catalog_fname):
    results = []
    filename_triangle = os.path.join(
        MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
    filename_star = os.path.join(
        MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
    with open(filename_triangle, 'rb') as f:
        triangle_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
    with open(filename_star, 'rb') as f:
        star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
    times = []
    star_identifier = StarIdentifier(
        planar_triangle_calculator=PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        ),
        triangle_catalog=triangle_catalog,
        star_catalog=star_catalog,
    )
    for row in input_data:
        start = timer()
        result = star_identifier.identify_stars(row)
        times.append(timer() - start)
        results.append([result])
        # plot_result(result)
        # create_image(result)

    print('Average time: ', np.sum(times)/len(times))
    return results


def plot_result(stars):
    res_x = 1920  # pixels
    res_y = 1440  # pixels
    stars = np.array(stars)
    txt = stars[:, 1]
    txt = txt.astype(int)
    x = stars[:, 5]
    y = stars[:, 6]

    fig, ax = plt.subplots()
    ax.scatter(y, x)
    ax.set_ylim(ymin=0, ymax=res_y)  # invert the axis
    ax.set_xlim(xmin=0, xmax=res_x)  # invert the axis
    ax.xaxis.tick_top()  # and move the X-Axis

    for i, txt in enumerate(txt):
        ax.annotate(txt, (y[i], x[i]))
    plt.show()


def create_image(stars):
    res_x = 1920  # pixels
    res_y = 1440  # pixels
    img = np.zeros((res_y, res_x), dtype='uint8')
    for star in stars:
        x = int(star[5])
        y = int(star[6])
        img[x, y] = 255
        img[x-1, y] = 255
        img[x+1, y] = 255
        img[x, y-1] = 255
        img[x, y+1] = 255

    Image.fromarray(img, mode='L').convert('1').transpose(
        Image.FLIP_TOP_BOTTOM).save('test.png')


class TestValidate:
    def test_one_scene_uv(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'), '1_scene_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 1
        assert in_scene_bad_verify == 0

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 1
        assert in_triangle_bad_verify == 0

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 1
        assert exact_bad_verify == 0

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 100.0

    def test_10_scenes_mag5_uv(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '10_scenes_mag_5_fov_10_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 10
        assert in_scene_bad_verify == 0

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 10
        assert in_triangle_bad_verify == 0

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 10
        assert exact_bad_verify == 0

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 100.0

    def test_100_scenes_mag5_uv(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '100_scenes_mag_5_fov_10_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 100
        assert in_scene_bad_verify == 0

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 100
        assert in_triangle_bad_verify == 0

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 99
        assert exact_bad_verify == 1

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 97.92

    def test_1000_scenes_mag4_uv_cat_mag5(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 982
        assert in_scene_bad_verify == 18

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 982
        assert in_triangle_bad_verify == 18

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 982
        assert exact_bad_verify == 18

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 99.69

    def test_1000_scenes_mag5_uv_cat_mag5(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 1000
        assert in_scene_bad_verify == 0

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 993
        assert in_triangle_bad_verify == 7

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 993
        assert exact_bad_verify == 7

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 99.17

    def test_1000_scenes_mag556_uv_cat_mag5(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 922
        assert in_scene_bad_verify == 78

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 904
        assert in_triangle_bad_verify == 96

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 904
        assert exact_bad_verify == 96

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 32.2

    @pytest.mark.skip('Too long test')
    def test_1000_scenes_mag556_uv_cat_mag6(self):
        input_data, expected = read_scene_uv(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag556_fov_10_uv')
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag6_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 984
        assert in_scene_bad_verify == 16

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 979
        assert in_triangle_bad_verify == 21

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 979
        assert exact_bad_verify == 21

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 95.47

        """
        Total time: 36m 41s
        Average time: 2.193535611206003
        """

    def test_1000_scenes_mag4_xy_cat_mag5(self):
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_4_fov_10_xy_scramble',
            focal_length, (res_x, res_y))
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 972
        assert in_scene_bad_verify == 28

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 929
        assert in_triangle_bad_verify == 71

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 929
        assert exact_bad_verify == 71

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 95.03

    def test_1000_scenes_mag5_xy_cat_mag5(self):
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_xy_scramble',
            focal_length, (res_x, res_y))
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 996
        assert in_scene_bad_verify == 4

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 961
        assert in_triangle_bad_verify == 39

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 960
        assert exact_bad_verify == 40

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 95.89

    def test_1000_scenes_mag5_xy_no_scramble_cat_mag5(self):
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            '1000_scenes_mag_5_fov_10_xy_no_scramble',
            focal_length, (res_x, res_y))
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 1000
        assert in_scene_bad_verify == 0

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 990
        assert in_triangle_bad_verify == 10

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 989
        assert exact_bad_verify == 11

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 95.79

    def test_esa_xy_cat_mag5(self):
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'esa_xy',
            focal_length, (res_x, res_y))
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 90
        assert in_scene_bad_verify == 10

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 90
        assert in_triangle_bad_verify == 10

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 90
        assert exact_bad_verify == 10

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 44.76

    @pytest.mark.skip('Too long test')
    def test_esa_xy_cat_mag6(self):
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'esa_xy',
            focal_length, (res_x, res_y))
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag6_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 66
        assert in_scene_bad_verify == 34

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 66
        assert in_triangle_bad_verify == 34

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 66
        assert exact_bad_verify == 34

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 56.51

        """
        Total time: 8m 18s
        Average time: 4.906043362510015
        """

    def test_tracker_scene_without_tracking_mode(self):
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            'tracker',
            focal_length, (res_x, res_y))
        result_verify = find_stars(
            input_data, 'triangle_catalog_mag5_fov10_full_area')

        in_scene_good_verify, in_scene_bad_verify = stars_in_scene(
            result_verify, expected)
        print('in scene good verify: ', in_scene_good_verify)
        print('in scene bad verify: ', in_scene_bad_verify)
        assert in_scene_good_verify == 12
        assert in_scene_bad_verify == 0

        in_triangle_good_verify, in_triangle_bad_verify = stars_in_triangle(
            result_verify, expected)
        print('in triangle good verify: ', in_triangle_good_verify)
        print('in triangle bad verify: ', in_triangle_bad_verify)
        assert in_triangle_good_verify == 12
        assert in_triangle_bad_verify == 0

        exact_good_verify, exact_bad_verify = exact_stars(
            result_verify, expected)
        print('exact good verify: ', exact_good_verify)
        print('exact bad verify: ', exact_bad_verify)
        assert exact_good_verify == 12
        assert exact_bad_verify == 0

        percent_verify = percent_stars_found(result_verify, expected)
        print('percent_identified_verify: ', percent_verify)
        assert percent_verify == 98.61


def stars_in_scene(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        try:
            scene_result = result[i][0]
            b = 0
            for s in scene_result:
                if s[1] not in expected[i] and s[1] != -1:
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad


def stars_in_triangle(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        try:
            res = result[i][0]
            b = 0
            for s in range(len(res)-2):  # res[:-2]:
                s1 = int(res[s][1])
                s2 = int(res[s+1][1])
                s3 = int(res[s+2][1])
                expected_triangle = [
                    expected[i][s],
                    expected[i][s+1],
                    expected[i][s+2],
                    -1,
                ]
                if not all([
                    s1 in expected_triangle,
                    s2 in expected_triangle,
                    s3 in expected_triangle
                ]):
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad


def exact_stars(result, expected):
    good = 0
    bad = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        try:
            scene_result = result[i][0]
            b = 0
            for s in range(len(scene_result)):
                if (scene_result[s][1] != expected[i][s] and
                        scene_result[s][1] != -1):
                    b += 1
            if b == 0:
                good += 1
            else:
                bad += 1
        except (AttributeError, TypeError, IndexError):
            bad += 1
    return good, bad


def percent_stars_found(result, expected):
    percent = 0
    for i in range(len(result)):
        assert len(result[i]) > 0
        p = 0
        try:
            scene_result = result[i][0]
            for s in range(len(scene_result)):
                if scene_result[s][1] == expected[i][s]:
                    p += 1
            p = p/len(scene_result)
        except (AttributeError, TypeError, IndexError):
            percent += 0
        percent += p
    percent = round(percent/len(result)*100, 2)
    return percent
