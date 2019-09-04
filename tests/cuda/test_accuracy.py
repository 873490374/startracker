import os

import numpy as np
import matplotlib.pyplot as plt

# noinspection PyPackageRequirements
import mock
# noinspection PyPackageRequirements
import pytest
from PIL import Image
from Quaternion import Quat
from astropy import units as u
from astropy.coordinates import Angle

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


images_path = os.path.join(MAIN_PATH, 'tests/images/')

# RA/DEC (J2000.0), z axis should be 0 degrees
expected_attitude = [
    # {'RA': '5h31m59.36s', 'DEC': '+2 05 41.3'},  # different z axis (non J2000)
    {'RA': '5h31m59.36s', 'DEC': '+2 05 41.3'},
    {'RA': '5h26m02.87s', 'DEC': '+2 35 45.2'},
    {'RA': '5h17m06.08s', 'DEC': '+2 41 45.1'},
    {'RA': '5h15m10.81s', 'DEC': '+3 05 51.7'},
    {'RA': '5h10m56.87s', 'DEC': '+3 06 41.7'},
    {'RA': '5h07m04.35s', 'DEC': '+3 50 02.8'},
    {'RA': '4h58m50.45s', 'DEC': '+3 20 03.3'},
]


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


@pytest.mark.cuda
class TestTracking:

    def test_tracking(self, star_tracker, image_processor):
        all_ = 0
        good = 0
        bad = 0
        not_recognized = 0
        attitude_not_found = 0

        sg = star_tracker.run()
        for i in range(1, 7):
            img_path = os.path.join(
                images_path, 'test_accuracy_{}.png'.format(i))
            img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
            img = img.convert('L')
            with mock.patch.object(
                    image_processor, 'get_image', return_value=img):

                stars, q = next(sg)
                a, g, b, n, att = validate(stars, q, expected_attitude[i])
                all_ += a
                good += g
                bad += b
                not_recognized += n
                attitude_not_found += att


def validate(stars, q, expected):
    all_ = 0
    good = 0
    bad = 0
    not_recognized = 0
    attitude_not_found = 0

    if not stars or q is None:
        attitude_not_found += 1
    else:
        print('')
        print('Stars ', stars)
        print('Quaternion =', q)
        if q is not None:
            q = Quat(q)
            print(q.equatorial)
            xx = (360 - q.ra) + 90
            # xx = q.ra
            print(xx)
            print(Angle('{}d'.format(xx)).to_string(unit=u.hour))
            print(q.dec)
            print(q.roll0)

        # plot_result(stars, 900, 900)
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
