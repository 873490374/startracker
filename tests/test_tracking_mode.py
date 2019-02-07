import os

import numpy as np

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.parallel.kvector_calculator_parallel import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracking_mode import TrackingMode
from program.utils import read_scene_xy


class TestTrackingMode:

    def test_tracking(self):
        tracker = TrackingMode(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=SENSOR_VARIANCE,
            ))
        kv_m = 2.83056279997e-07
        kv_q = -2.03606250703e-07
        camera_fov = 10  # degrees
        focal_length = 0.5 / np.tan(np.deg2rad(camera_fov) / 2)  # pixels
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        input_data, expected = read_scene_xy(
            os.path.join(MAIN_PATH, 'tests/scenes'),
            'tracker', focal_length, (res_x, res_y))
        filename_triangle = os.path.join(
            MAIN_PATH,
            'tests/catalog/triangle_catalog_mag5_fov10_full_area.csv')
        filename_star = os.path.join(
            MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
        with open(filename_triangle, 'rb') as f:
            triangle_catalog = np.genfromtxt(
                f, dtype=np.float64, delimiter=',')
        with open(filename_star, 'rb') as f:
            star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        star_identifier = StarIdentifier(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=SENSOR_VARIANCE
            ),
            kvector_calculator=KVectorCalculator(kv_m, kv_q),
            triangle_catalog=triangle_catalog,
            star_catalog=star_catalog,
            verify_stars_flag=True
        )
        result = star_identifier.identify_stars(input_data[0])
        # time
        for i in range(1, len(input_data)):
            result = tracker.track(input_data[i], result)
            for x in result:
                try:
                    assert x[1] in expected[i]  # [int(x[0])]
                except AssertionError:
                    print('Lost attitude, frame ', i)
                try:
                    assert x[1] == expected[i][int(x[0])]
                except AssertionError:
                    print('not the same star, frame ', i)
            result = star_identifier.identify_stars(input_data[i])

        #  1st star identifier
        #  then tracker
