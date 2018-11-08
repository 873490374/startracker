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
        res_x = 491
        res_y = 491
        pixel_size = 1
        focal_length = FOCAL_LENGTH * 491
        a_roi = 5
        i_threshold = 250
        principal_point = (0.5 * res_x, 0.5 * res_y)
        img_path = os.path.join(self.images_path, '2.jpg')
        image = Image.open(img_path)
        centroid_calculator = CentroidCalculator(
            pixel_size,
            focal_length,
            a_roi,
            i_threshold,
            principal_point
        )
        I = ImageProcessor(
            CameraConnector(), centroid_calculator).image_to_matrix(image)
        I.setflags(write=1)
        for i in range(10):
            start_time = timer()
            list_of_stars = centroid_calculator.calculate_centroids(I)
            print(timer() - start_time)
            assert 92 == len(list_of_stars)
            uv = list_of_stars[0].unit_vector
            assert np.isclose(0.0698713971976980, uv[0], atol=0.00000000000001)
            assert np.isclose(0.9975510033514716, uv[1], atol=0.00000000000001)
            assert np.isclose(0.0031596781661251, uv[2], atol=0.00000000000001)
            uv = list_of_stars[91].unit_vector
            assert np.isclose(0.7298603946127173, uv[0], atol=0.00000000000001)
            assert np.isclose(0.6835919812794351, uv[1], atol=0.00000000000001)
            assert np.isclose(0.0024098768070093, uv[2], atol=0.00000000000001)
