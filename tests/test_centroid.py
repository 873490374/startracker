import datetime

import numpy as np
import os
from PIL import Image
from timeit import default_timer as timer

from program.const import MAIN_PATH
from program.star import StarUV
from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator
from program.tracker.image_processor import ImageProcessor

"""
Time: ~1s
"""


class TestCentroid:
    pixel_size = 5
    focal_length = 7
    a_roi = 5
    i_threshold = 250
    images_path = os.path.join(MAIN_PATH, 'tests/images/stars/')

    def test_centroid_jpg_2(self):
        img_path = os.path.join(self.images_path, '2.jpg')
        image = Image.open(img_path)
        centroid_calculator = CentroidCalculator(
            self.pixel_size,
            self.focal_length,
            self.a_roi,
            self.i_threshold,
        )
        I = ImageProcessor(
            CameraConnector(), centroid_calculator).image_to_matrix(image)
        for i in range(10):
            start_time = timer()
            list_of_stars = centroid_calculator.calculate_centroids(I)
            print(timer() - start_time)
            assert 92 == len(list_of_stars)
            uv = list_of_stars[0].unit_vector
            assert np.isclose(0.06987140, uv[0], atol=0.001)
            assert np.isclose(0.99755100, uv[1], atol=0.0001)
            assert np.isclose(0.00315968, uv[2], atol=0.0001)
            uv = list_of_stars[91].unit_vector
            assert np.isclose(0.72986039, uv[0], atol=0.0001)
            assert np.isclose(0.68359198, uv[1], atol=0.0001)
            assert np.isclose(0.00240988, uv[2], atol=0.0001)
