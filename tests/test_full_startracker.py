import os

import numpy as np
import matplotlib.pyplot as plt

# noinspection PyPackageRequirements
import mock
from PIL import Image

from program.const import FOCAL_LENGTH, SENSOR_VARIANCE, MAIN_PATH
from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor
from program.tracker.main_program import StarTracker
from program.tracker.orientation_finder import OrientationFinder
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.quest import QuestCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker


class TestFullStarTracker:

    def test_full_startracker(self):
        res_x = 900  # pixels
        res_y = 900  # pixels
        pixel_size = 1
        focal_length = FOCAL_LENGTH * res_x
        a_roi = 5
        c_roi = 10
        i_threshold = 150
        mag_threshold = 160
        star_mag_pix = 14
        principal_point = (0.5 * res_x, 0.5 * res_y)

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

        catalog_fname = 'triangle_catalog_mag5_fov10_full_area'
        filename_triangle = os.path.join(
            MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
        filename_star = os.path.join(
            MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
        with open(filename_triangle, 'rb') as f:
            triangle_catalog = np.genfromtxt(
                f, dtype=np.float64, delimiter=',')
        with open(filename_star, 'rb') as f:
            star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        planar_triangle_calc = PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        )
        st = StarTracker(
            image_processor=image_processor,
            star_identifier=StarIdentifier(
                planar_triangle_calculator=planar_triangle_calc,
                triangle_catalog=triangle_catalog,
                star_catalog=star_catalog,
            ),
            orientation_finder=OrientationFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calc),
            tracking_mode_enabled=False,
        )

        sg = st.run()
        for i in range(14):
            images_path = os.path.join(MAIN_PATH, 'tests/images/')
            img_path = os.path.join(images_path, 'test_full_{}.png'.format(i))
            img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
            with mock.patch.object(
                    image_processor, 'get_image', return_value=img):

                stars, q = next(sg)
                print('Stars:')
                if not stars:
                    continue
                [print(int(s[0]), '\t ',  int(s[1]), '\t',
                       'uv:', s[2], s[3], s[4], 'xy:', s[5], s[6])
                    for s in stars]
                print([int(s[1]) for s in stars])
                print('Quaternion =', q)

                # for s in range(len(stars)):
                #     assert stars[s][1] == expected[0][s] or stars[s][1] == -1
                plot_result(stars, res_x, res_y)

    def test_full_startracker_moon(self):
        res_x = 900  # pixels
        res_y = 900  # pixels
        pixel_size = 1
        focal_length = FOCAL_LENGTH * res_x
        a_roi = 5
        c_roi = 10
        i_threshold = 150
        mag_threshold = 160
        star_mag_pix = 14
        principal_point = (0.5 * res_x, 0.5 * res_y)

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

        catalog_fname = 'triangle_catalog_mag5_fov10_full_area'
        filename_triangle = os.path.join(
            MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
        filename_star = os.path.join(
            MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
        with open(filename_triangle, 'rb') as f:
            triangle_catalog = np.genfromtxt(
                f, dtype=np.float64, delimiter=',')
        with open(filename_star, 'rb') as f:
            star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        planar_triangle_calc = PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        )
        st = StarTracker(
            image_processor=image_processor,
            star_identifier=StarIdentifier(
                planar_triangle_calculator=planar_triangle_calc,
                triangle_catalog=triangle_catalog,
                star_catalog=star_catalog,
            ),
            orientation_finder=OrientationFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calc),
            tracking_mode_enabled=False,
        )

        sg = st.run()
        images_path = os.path.join(MAIN_PATH, 'tests/images/')
        img_path = os.path.join(images_path, 'test_full_moon.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        with mock.patch.object(
                image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            print('Stars:')
            [print(int(s[0]), '\t ',  int(s[1]), '\t',
                   'uv:', s[2], s[3], s[4], 'xy:', s[5], s[6])
                for s in stars]
            print([int(s[1]) for s in stars])
            print('Quaternion =', q)
            plot_result(stars, res_x, res_y)

    def test_full_startracker_sun(self):
        res_x = 900  # pixels
        res_y = 900  # pixels
        pixel_size = 1
        focal_length = FOCAL_LENGTH * res_x
        a_roi = 5
        c_roi = 10
        i_threshold = 150
        mag_threshold = 160
        star_mag_pix = 14
        principal_point = (0.5 * res_x, 0.5 * res_y)

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

        catalog_fname = 'triangle_catalog_mag5_fov10_full_area'
        filename_triangle = os.path.join(
            MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
        filename_star = os.path.join(
            MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
        with open(filename_triangle, 'rb') as f:
            triangle_catalog = np.genfromtxt(
                f, dtype=np.float64, delimiter=',')
        with open(filename_star, 'rb') as f:
            star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        planar_triangle_calc = PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        )
        st = StarTracker(
            image_processor=image_processor,
            star_identifier=StarIdentifier(
                planar_triangle_calculator=planar_triangle_calc,
                triangle_catalog=triangle_catalog,
                star_catalog=star_catalog,
            ),
            orientation_finder=OrientationFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calc),
            tracking_mode_enabled=False,
        )

        sg = st.run()
        images_path = os.path.join(MAIN_PATH, 'tests/images/')
        img_path = os.path.join(images_path, 'test_full_sun.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        with mock.patch.object(
                image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            print('Stars:')
            [print(int(s[0]), '\t ',  int(s[1]), '\t',
                   'uv:', s[2], s[3], s[4], 'xy:', s[5], s[6])
                for s in stars]
            print([int(s[1]) for s in stars])
            print('Quaternion =', q)
            plot_result(stars, res_x, res_y)

    def test_full_startracker_brightness(self):
        res_x = 900  # pixels
        res_y = 900  # pixels
        pixel_size = 1
        focal_length = FOCAL_LENGTH * res_x
        a_roi = 5
        c_roi = 10
        i_threshold = 160
        mag_threshold = 180
        star_mag_pix = 14
        principal_point = (0.5 * res_x, 0.5 * res_y)

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

        catalog_fname = 'triangle_catalog_mag5_fov10_full_area'
        filename_triangle = os.path.join(
            MAIN_PATH, 'tests/catalog/{}.csv'.format(catalog_fname))
        filename_star = os.path.join(
            MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
        with open(filename_triangle, 'rb') as f:
            triangle_catalog = np.genfromtxt(
                f, dtype=np.float64, delimiter=',')
        with open(filename_star, 'rb') as f:
            star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        planar_triangle_calc = PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE
        )
        st = StarTracker(
            image_processor=image_processor,
            star_identifier=StarIdentifier(
                planar_triangle_calculator=planar_triangle_calc,
                triangle_catalog=triangle_catalog,
                star_catalog=star_catalog,
            ),
            orientation_finder=OrientationFinder(
                quest_calculator=QuestCalculator(),
                star_catalog=star_catalog,
            ),
            tracker=Tracker(
                planar_triangle_calculator=planar_triangle_calc),
            tracking_mode_enabled=False,
        )

        sg = st.run()
        images_path = os.path.join(MAIN_PATH, 'tests/images/')
        img_path = os.path.join(images_path, 'test_full_brightness.png')
        img = Image.open(img_path).transpose(Image.FLIP_TOP_BOTTOM)
        with mock.patch.object(
                image_processor, 'get_image', return_value=img):

            stars, q = next(sg)
            print('Stars:')
            [print(int(s[0]), '\t ',  int(s[1]), '\t',
                   'uv:', s[2], s[3], s[4], 'xy:', s[5], s[6])
                for s in stars]
            print([int(s[1]) for s in stars])
            print('Quaternion =', q)
            plot_result(stars, res_x, res_y)


def plot_result(stars, res_x, res_y):
    stars = np.array(stars)
    txt = stars[:, 1]
    txt = txt.astype(int)
    x = stars[:, 5]
    y = stars[:, 6]

    fig, ax = plt.subplots()
    ax.scatter(y, x)
    ax.set_xlim(xmin=0, xmax=res_x)
    ax.set_ylim(ymin=0, ymax=res_y)

    for i, txt in enumerate(txt):
        ax.annotate(txt, (y[i], x[i]))
    plt.show()


expected = [
    [28816, 28910, 28574, 28325, 25923, 26563, 27288, 28103, 27366],
    [25923, 28910, 28574, 28325, 26563, 27288, 28103, 27366, -1, 26235],
    [28816, 28910, 29735, 28574, 30324, 27288, 28103, 27366],
]
