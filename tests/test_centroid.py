import os
from PIL import Image

from program.tracker.centroid import CentroidCalculator

#
# def centroid_jpg2_wrapper():
#     pixel_size = 5
#     focal_length = 7
#     a_roi = 5
#     i_threshold = 250
#     img_name = '2.jpg'
#     path = 'images/stars/'
#     image = Image.open(path + img_name)
#
#     centroid_calculator = CentroidCalculator(
#         pixel_size, focal_length, a_roi, i_threshold
#     )
#     return centroid_calculator.calculate_centroids(image)
#
#
# class TestCentroid:
#
#     def test_centroid_jpg_2_benchmark(self, benchmark):
#
#         list_of_stars = benchmark(centroid_jpg2_wrapper)
#         assert len(list_of_stars) == 92


class TestCentroid:
    pixel_size = 5
    focal_length = 7
    a_roi = 5
    i_threshold = 250
    p = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(p, 'images/stars/')

    def test_centroid_jpg_2(self):
        img_name = '2.jpg'
        image = Image.open(self.path + img_name)
        centroid_calculator = CentroidCalculator(
            self.pixel_size,
            self.focal_length,
            self.a_roi,
            self.i_threshold,
        )
        list_of_stars = centroid_calculator.calculate_centroids(image)
        assert len(list_of_stars) == 92
        assert len(list_of_stars[0]) == 3
        print(list_of_stars[0])
