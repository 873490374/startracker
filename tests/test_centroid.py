from PIL import Image
from pytest_benchmark.plugin import benchmark

from centroid import CentroidCalculator

def bb(integra):
    return 10*integra

class TestCentroid:

    def test_centroid_jpg_2(self, benchmark):
        pixel_size = 5
        focal_length = 7
        a_roi = 5
        i_threshold = 250
        img_name = '2.jpg'
        path = 'images/stars/'
        image = Image.open(path+img_name)
        # centroid_calculator = CentroidCalculator(
        #     pixel_size, focal_length, a_roi, i_threshold
        # )
        # star_list = benchmark(centroid_calculator.calculate_centroids(image))
        # print(star_list)

        x = benchmark(bb(88))
        print(x)
