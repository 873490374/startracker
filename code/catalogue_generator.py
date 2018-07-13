import numpy as np

from code.planar_triangle import PlanarTriangle
from testing.scripts.simulator import StarCatalog

MAX_MAGNITUDE = 999
CAMERA_FOV = 999
sensor_variance = 1


class StarPosition:
    def __init__(self, star_id, magnitude, right_ascension, declination):
        self.id = star_id
        self.magnitude = magnitude
        self.right_ascension = right_ascension
        self.declination = declination

    def __str__(self):
        return "{}, {}, {}, {}".format(
            self.id, self.magnitude, self.right_ascension, self.declination
        )


class StarUV:
    def __init__(self, star_id, magnitude, unit_vector):
        self.id = star_id
        self.magnitude = magnitude
        self.unit_vector = unit_vector

    def __str__(self):
        return "{}, {}, {}".format(
            self.id, self.magnitude, self.unit_vector,
        )


class CatalogueTriangle:
    def __init__(self, star1_id, star2_id, star3_id, area, polar_moment):
        self.star1_id = star1_id
        self.star2_id = star2_id
        self.star3_id = star3_id
        self.area = area
        self.polar_moment = polar_moment


class CatalogueGenerator:

    def generate_triangles(self) -> [CatalogueTriangle]:
        converted_start = []
        triangle_catalogue = []

        stars = self.read_catalogue_stars()
        return
        for s in stars:
            star = self.convert(s)
            converted_start.append(star)
            print(star)

        for s1 in converted_start:
            for s2 in converted_start:
                for s3 in converted_start:
                    if self.is_valid(s1, s2, s3):
                        triangle = PlanarTriangle()
                        triangle.calculate_triangle(
                            s1.unit_vector,
                            s2.unit_vector,
                            s3.unit_vector,
                            sensor_variance
                        )
                        triangle_catalogue.append(
                            CatalogueTriangle(
                                s1, s2, s3, triangle.A, triangle.J))
                        print(triangle)

        return triangle_catalogue

    def read_catalogue_stars(self, ) -> [StarPosition]:
        stars = []
        import os

        catalog = StarCatalog(
            '{}/../testing/data/hip_main.dat'.format(os.getcwd())).catalog
        for row in catalog:
            print(row)
            s = StarPosition(
                row[1],
                row[5],
                row[8],
                row[9]
            )
            stars.append(s)
            print(s)
            break
        return stars

    def convert(self, star: StarPosition) -> StarUV:
        """ Convert star positions to unit vector."""
        alpha = star.right_ascension
        delta = star.declination
        return StarUV(
            star_id=star.id,
            magnitude=star.magnitude,
            unit_vector=np.array([
                np.cos(alpha) * np.cos(delta),
                np.sin(alpha) * np.cos(delta),
                np.sin(delta)
            ]).T
        )

    def is_valid(self, s1: StarUV, s2: StarUV, s3: StarUV) -> bool:
        print(s1, s2, s3)
        return all(
            s1.magnitude <= MAX_MAGNITUDE and
            s2.magnitude <= MAX_MAGNITUDE and
            s3.magnitude <= MAX_MAGNITUDE and
            s1.unit_vector.T * s2.unit_vector >= np.cos(CAMERA_FOV) and
            s1.unit_vector.T * s3.unit_vector >= np.cos(CAMERA_FOV) and
            s2.unit_vector.T * s3.unit_vector >= np.cos(CAMERA_FOV) and
            s1 is not s2 and s2 is not s3 and s1 is not s3
        )


generator = CatalogueGenerator()
generator.generate_triangles()
