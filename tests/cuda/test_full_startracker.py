import os

import numpy as np
import matplotlib.pyplot as plt

# noinspection PyPackageRequirements
import mock
# noinspection PyPackageRequirements
import pytest
from PIL import Image

from program.const import MAIN_PATH
from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor
from program.tracker.main_program import StarTracker
from program.tracker.attitude_finder import AttitudeFinder
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.quest import QuestCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker
from tests.expected_results_full_startracker import (
    expected_full,
    expected_moon,
    expected_sun,
    expected_brightness,
)

images_path = os.path.join(MAIN_PATH, 'tests/images/')


@pytest.fixture
def camera_fov():
    return 10


@pytest.fixture
def cos_camera_fov(camera_fov):
    return np.cos(np.deg2rad(camera_fov))


@pytest.fixture
def focal_length_normalized(camera_fov):
    return 0.5 / np.tan(np.deg2rad(camera_fov) / 2)


@pytest.fixture
def sensor_variance():
    return 270e-6 / 10


@pytest.fixture
def res_x():
    return 900


@pytest.fixture
def res_y():
    return 900


@pytest.fixture
def pixel_size():
    return 1


@pytest.fixture
def focal_length(focal_length_normalized, res_x):
    return focal_length_normalized * res_x


@pytest.fixture
def a_roi():
    return 5


@pytest.fixture
def c_roi():
    return 10


@pytest.fixture
def i_threshold():
    return 150


@pytest.fixture
def mag_threshold():
    return 160


@pytest.fixture
def star_mag_pix():
    return 14


@pytest.fixture
def principal_point(res_x, res_y):
    return 0.5 * res_x, 0.5 * res_y


@pytest.fixture
def centroid_calculator(
        pixel_size, focal_length, a_roi, c_roi,
        i_threshold, mag_threshold, star_mag_pix, principal_point):
    return CentroidCalculator(
        pixel_size,
        focal_length,
        a_roi,
        c_roi,
        i_threshold,
        mag_threshold,
        star_mag_pix,
        principal_point
    )


@pytest.fixture
def image_processor(centroid_calculator):
    return ImageProcessor(CameraConnector(), centroid_calculator)


@pytest.fixture
def triangle_catalog():
    catalog_fname = 'triangle_catalog_mag5_fov10_full_area'
    filename_triangle = os.path.join(
        MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
    with open(filename_triangle, 'rb') as f:
        triangle_catalog = np.genfromtxt(
            f, dtype=np.float64, delimiter=',')
    return triangle_catalog


@pytest.fixture
def star_catalog():
    filename_star = os.path.join(
        MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
    with open(filename_star, 'rb') as f:
        star_catalog = np.genfromtxt(
            f, dtype=np.float64, delimiter=',')
    return star_catalog


@pytest.fixture
def planar_triangle_calculator(sensor_variance):
    return PlanarTriangleCalculator(
        sensor_variance=sensor_variance
    )


@pytest.fixture
def star_tracker(
        image_processor, planar_triangle_calculator,
        triangle_catalog, star_catalog):
    return StarTracker(
        image_processor=image_processor,
        star_identifier=StarIdentifier(
            planar_triangle_calculator=planar_triangle_calculator,
            triangle_catalog=triangle_catalog,
            star_catalog=star_catalog,
        ),
        attitude_finder=AttitudeFinder(
            quest_calculator=QuestCalculator(),
            star_catalog=star_catalog,
        ),
        tracker=Tracker(
            planar_triangle_calculator=planar_triangle_calculator),
        tracking_mode_enabled=False,
    )


class TestFullStarTracker:

    def test_full_startracker(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        for i in range(14):
            img_path = os.path.join(
                images_path, 'test_full_{}.png'.format(i))
            img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert('L')
            with mock.patch.object(
                    image_processor, 'get_image', return_value=img):

                stars, q = next(sg)
                a, g, b, n, att = validate(stars, q, expected_full[i])
                all_ += a
                good += g
                bad += b
                not_recognized += n
                attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))

    def test_full_startracker_moon(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        img_path = os.path.join(images_path, 'test_full_moon.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert('L')
        with mock.patch.object(image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            a, g, b, n, att = validate(stars, q, expected_moon)
            all_ += a
            good += g
            bad += b
            not_recognized += n
            attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))

    def test_full_startracker_sun(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        img_path = os.path.join(images_path, 'test_full_sun.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert('L')
        with mock.patch.object(image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            a, g, b, n, att = validate(stars, q, expected_sun)
            all_ += a
            good += g
            bad += b
            not_recognized += n
            attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))

    def test_full_startracker_brightness(
            self, pixel_size, focal_length, a_roi, c_roi, star_mag_pix,
            principal_point, planar_triangle_calculator,
            triangle_catalog, star_catalog):
        i_threshold = 160
        mag_threshold = 180

        centroid_calculator = CentroidCalculator(
            pixel_size,
            focal_length,
            a_roi,
            c_roi,
            i_threshold,
            mag_threshold,
            star_mag_pix,
            principal_point
        )
        image_processor = ImageProcessor(
            CameraConnector(), centroid_calculator)

        st = StarTracker(
            image_processor=image_processor,
            star_identifier=StarIdentifier(
                planar_triangle_calculator=planar_triangle_calculator,
                triangle_catalog=triangle_catalog,
                star_catalog=star_catalog,
            ),
            attitude_finder=AttitudeFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calculator),
            tracking_mode_enabled=False,
        )
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = st.run()
        img_path = os.path.join(images_path, 'test_full_brightness.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.convert('L')
        with mock.patch.object(
                image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            a, g, b, n, att = validate(stars, q, expected_brightness)
            all_ += a
            good += g
            bad += b
            not_recognized += n
            attitude_not_found += att

        print('All: {}'.format(all_))
        print('Good: {}'.format(good))
        print('Bad: {}'.format(bad))
        print('Not recognized: {}'.format(not_recognized))
        print('Attitude not found: {}'.format(attitude_not_found))


def validate(stars, q, expected):
    all_ = 0
    good = 0
    bad = 0
    not_recognized = 0
    attitude_not_found = 0

    if not stars or q is None:
        attitude_not_found += 1
    if not stars:
        not_recognized_in_scene = sum(
            [1 for s in expected if s['id_cat'] != -1])
        all_ += not_recognized_in_scene
        not_recognized += not_recognized_in_scene
    else:
        for es in expected:
            if es['id_cat'] != -1:
                for s in stars:
                    if (
                            np.isclose(s[5], es['x'], atol=0.00001) and
                            np.isclose(s[6], es['y'], atol=0.00001)):
                        all_ += 1
                        if s[1] == es['id_cat']:
                            good += 1
                        elif s[1] == -1:
                            not_recognized += 1
                        else:
                            bad += 1

        print('')
        print('Quaternion =', q)

        # plot_result(stars, res_x(), res_y())
    return all_, good, bad, not_recognized, attitude_not_found


def plot_result(stars, res_x_, res_y_):
    stars = np.array(stars)
    txt = stars[:, 1]
    txt = txt.astype(int)
    x = stars[:, 5]
    y = stars[:, 6]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlim(xmin=0, xmax=res_x_)
    ax.set_ylim(ymin=0, ymax=res_y_)

    for i, txt in enumerate(txt):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()
