import os

import numpy as np
from timeit import default_timer as timer

import pytest

from program.const import SENSOR_VARIANCE, MAIN_PATH
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.tracker.tracker import Tracker
from program.utils import read_scene_xy


@pytest.mark.skip('Needed better test scenes, those are changing too fast')
class TestTracker:

    def test_tracking(self):
        tracker = Tracker(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=SENSOR_VARIANCE,
            ))
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
            triangle_catalog=triangle_catalog,
            star_catalog=star_catalog,
        )
        for aaa in range(10):
            times = []
            lost_attitude_known = 0
            lost_attitude_unknown = 0
            scrambled_stars = 0
            result = star_identifier.identify_stars(input_data[0])
            for i in range(1, len(input_data)):
                # print('Frame ', i)
                # print([int(star[1]) for star in result])
                if not result or len((set([int(star[1]) for star in result]))) < 3:
                    # print('new star search')
                    start = timer()
                    result = star_identifier.identify_stars(input_data[i])
                    times.append(timer() - start)
                    # print('**** Lost attitude, frame ', i, '****')
                    lost_attitude_known += 1
                    continue
                start = timer()
                # print('tracking mode')
                result = tracker.track(input_data[i], result)
                # result = star_identifier.identify_stars(input_data[i])
                times.append(timer() - start)
                for s in result:
                    try:
                        assert s[1] in expected[i] or s[1] == -1
                    except AssertionError:
                        # print('**** Lost attitude not knowing, frame ', i)
                        lost_attitude_unknown += 1
                        break
                    try:
                        assert s[1] == expected[i][int(s[0])] or s[1] == -1
                    except AssertionError:
                        # print('**** Stars scrambled, frame ', i)
                        scrambled_stars += 1
                        break
            print('Average time: ', np.sum(times) / len(times))

            print('Scrambled stars: ', scrambled_stars)
            print('Lost attitude knowing: ', lost_attitude_known)
            print('Lost attitude not knowing: ', lost_attitude_unknown)
            print('Correctly: ', len(input_data) -
                  scrambled_stars -
                  lost_attitude_known -
                  lost_attitude_unknown)

        # 0,03833982454571841 / 0,015765512875077548 = 2,4319
        # 0,015765512875077548 / 0,03833982454571841 = 0,4112
