import numpy as np
import os
from PIL import Image
from timeit import default_timer as timer

from program.const import MAIN_PATH, FOCAL_LENGTH
from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor


class TestCentroid:
    images_path = os.path.join(MAIN_PATH, 'tests/images/stars/')

    def test_centroid_jpg_2(self):
        img_path = os.path.join(self.images_path, '2.jpg')
        image = Image.open(img_path)
        res_x = 691
        res_y = 691
        pixel_size = 1
        focal_length = FOCAL_LENGTH * res_x
        a_roi = 5
        i_threshold = 250
        principal_point = (0.5 * res_x, 0.5 * res_y)

        centroid_calculator = CentroidCalculator(
            pixel_size,
            focal_length,
            a_roi,
            i_threshold,
            principal_point
        )
        image_matrix = ImageProcessor(
            CameraConnector(), centroid_calculator).image_to_matrix(image)
        assert (res_x, res_y) == image_matrix.shape
        for i in range(10):
            start_time = timer()
            list_of_stars = centroid_calculator.calculate_centroids(
                image_matrix)
            print(timer() - start_time)
            assert 92 == len(list_of_stars)
            uv = list_of_stars[0].unit_vector
            assert np.isclose(-0.07936259453726248, uv[0], atol=1.e-10)
            assert np.isclose(0.02435176834680778, uv[1], atol=1.e-10)
            assert np.isclose(0.9965483279634247, uv[2], atol=1.e-10)
            uv = list_of_stars[91].unit_vector
            assert np.isclose(0.019888234048011138, uv[0], atol=1.e-10)
            assert np.isclose(0.013084809683807373, uv[1], atol=1.e-10)
            assert np.isclose(0.9997165827883371, uv[2], atol=1.e-10)
