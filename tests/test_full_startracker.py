import os

import numpy as np
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
        res_y = 1920  # pixels
        res_x = 1440  # pixels
        pixel_size = 1
        focal_length = FOCAL_LENGTH * res_y
        a_roi = 5
        c_roi = 15
        i_threshold = 200
        principal_point = (0.5 * res_x, 0.5 * res_y)

        centroid_calculator = CentroidCalculator(
            pixel_size,
            focal_length,
            a_roi,
            c_roi,
            i_threshold,
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

        images_path = os.path.join(MAIN_PATH, 'tests/images/stars/')
        img_path = os.path.join(images_path, '1.png')

        with mock.patch.object(
                image_processor, 'get_image',
                return_value=Image.open(img_path)):
            sg = st.run()
            stars, q = next(sg)
            print('Stars:')
            [print(int(s[0]), '\t',  int(s[1]), '\t', s[2], s[3], s[4]) for s in stars]
            print('Quaternion =', q)
