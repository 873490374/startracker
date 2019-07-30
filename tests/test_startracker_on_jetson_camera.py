import os

import numpy as np
import matplotlib.pyplot as plt
import pytest

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
    return 2592


@pytest.fixture
def res_y():
    return 1458


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
    return 20


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
@pytest.mark.skip('Only ot be tested with camera on test bed')
class TestStartrackerJetson:

    def test_full_startracker_jetson(
            self, star_tracker, res_x, res_y):
        sg = star_tracker.run()
        for i in range(10):
            stars, q = next(sg)
            [print(int(x[0]), '\t', int(x[1]), '\t', int(x[5]), '\t', int(x[6])) for x in stars]
            print(q)
            plot_result(stars, res_x, res_y)


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
