from PIL import Image

from startracker.centroid import CentroidCalculator


def centroid_jpg2_wrapper():
    pixel_size = 5
    focal_length = 7
    a_roi = 5
    i_threshold = 250
    img_name = '2.jpg'
    path = 'images/stars/'
    image = Image.open(path + img_name)

    centroid_calculator = CentroidCalculator(
        pixel_size, focal_length, a_roi, i_threshold
    )
    return centroid_calculator.calculate_centroids(image)


class TestCentroid:

    def test_centroid_jpg_2_benchmark(self, benchmark):

        list_of_stars = benchmark(centroid_jpg2_wrapper)
        assert len(list_of_stars) == 92
