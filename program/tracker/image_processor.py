import numpy as np

from program.tracker.camera import CameraConnector
from program.tracker.centroid import CentroidCalculator


class ImageProcessor:
    def __init__(
            self,
            camera_connector: CameraConnector,
            centroid_calculator: CentroidCalculator,
    ):
        self.camera_conn = camera_connector
        self.centroid_calc = centroid_calculator

    def get_image_matrix(self) -> np.ndarray:
        image = self.get_image()
        return self.image_to_matrix(image)

    def get_image(self) -> np.ndarray:
        return self.camera_conn.get_image()

    @staticmethod
    def image_to_matrix(image: np.ndarray) -> np.ndarray:
        return np.asarray(image)

    def get_image_star_vectors(self) -> np.ndarray:
        img_matrix = self.get_image_matrix()
        return self.get_star_vectors(img_matrix)

    def get_star_vectors(self, img_matrix: np.ndarray) -> np.ndarray:
        return self.centroid_calc.calculate_centroids(img_matrix)
