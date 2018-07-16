import numpy as np

from progress.bar import Bar

from program.classes import StarUV
from program.tracker.planar_triangle import PlanarTriangle


class StarIdentifier:

    def __init__(self, sensor_variance, max_magnitude, camera_fov):
        self.sensor_variance = sensor_variance
        self.max_magnitude = max_magnitude
        self.camera_fov = camera_fov

    def identify_stars(self, stars_list: [StarUV]):
        classify_bar = Bar(
            'Building planar triangle catalogue',
            max=len(stars_list))
        for s1 in stars_list:
            classify_bar.next()
            for s2 in stars_list:
                for s3 in stars_list:
                    if is_valid(
                            s1, s2, s3, self.max_magnitude, self.camera_fov):
                        triangle = PlanarTriangle()
                        triangle.calculate_triangle(
                            s1.unit_vector,
                            s2.unit_vector,
                            s3.unit_vector,
                            self.sensor_variance
                        )
                        CatalogueTriangle(
                            s1.id, s2.id, s3.id, triangle.A,
                            triangle.J)


def is_valid(
        s1: StarUV, s2: StarUV, s3: StarUV,
        max_magnitude: int, camera_fov: int) -> bool:
    return all([
        s1.magnitude <= max_magnitude,
        s2.magnitude <= max_magnitude,
        s3.magnitude <= max_magnitude,
        s1.id is not s2.id, s2.id is not s3.id, s1.id is not s3,
        not any(s1.unit_vector.T * s2.unit_vector >= np.cos(camera_fov)),
        not any(s1.unit_vector.T * s3.unit_vector >= np.cos(camera_fov)),
        not any(s2.unit_vector.T * s3.unit_vector >= np.cos(camera_fov)),
    ])
