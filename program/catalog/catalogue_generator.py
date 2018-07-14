import numpy as np
from progress.bar import Bar

from program.const import sensor_variance, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.planar_triangle import PlanarTriangle
from program.validation.scripts.simulator import StarCatalog


class StarPosition:
    def __init__(
            self, star_id: int, magnitude: float, right_ascension: float,
            declination: float):
        self.id = star_id
        self.magnitude = magnitude
        self.right_ascension = right_ascension
        self.declination = declination

    def __str__(self):
        return "{}, {}, {}, {}".format(
            self.id, self.magnitude, self.right_ascension, self.declination
        )


class StarUV:
    def __init__(
            self, star_id: int, magnitude: float, unit_vector: np.ndarray):
        self.id = star_id
        self.magnitude = magnitude
        self.unit_vector = unit_vector

    def __str__(self):
        return "{}, {}, {}".format(
            self.id, self.magnitude, self.unit_vector,
        )


class CatalogueTriangle:
    def __init__(
            self, star1_id: int, star2_id: int, star3_id: int, area: np.double,
            polar_moment: np.double):
        self.star1_id = star1_id
        self.star2_id = star2_id
        self.star3_id = star3_id
        self.area = area
        self.polar_moment = polar_moment

    def __str__(self):
        return "{}, {}, {}, {}, {}".format(
            self.star1_id, self.star2_id, self.star3_id,
            self.area, self.polar_moment,
        )


class CatalogueGenerator:

    def generate_triangles(self) -> [CatalogueTriangle]:
        converted_start = []
        triangle_catalogue = set()

        stars = self.read_catalogue_stars()
        for s in stars:
            star = self.convert(s)
            converted_start.append(star)
        print(len(converted_start))
        classify_bar = Bar('Classify', max=len(converted_start))
        for s1 in converted_start:
            i = 1
            # print(s1.id)
            for s2 in converted_start:
                # print(s2.id)
                for s3 in converted_start:
                    # print(s3.id)
                    if self.is_valid(s1, s2, s3):
                        triangle = PlanarTriangle()
                        triangle.calculate_triangle(
                            s1.unit_vector,
                            s2.unit_vector,
                            s3.unit_vector,
                            sensor_variance
                        )
                        triangle_catalogue.add(
                            CatalogueTriangle(
                                s1.id, s2.id, s3.id, triangle.A, triangle.J))
                        print(CatalogueTriangle(
                                s1.id, s2.id, s3.id, triangle.A, triangle.J))
            i += 1
            classify_bar.next()
        classify_bar.finish()

        return triangle_catalogue

    def read_catalogue_stars(self) -> [StarPosition]:
        stars = []
        import os

        catalog = StarCatalog(
            '{}/program/validation/data/hip_main.dat'.format(os.getcwd())).catalog
        for row in catalog.itertuples():
            s = StarPosition(
                row[2],
                row[6],
                row[9],
                row[10],
            )
            stars.append(s)
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
        return all([
            s1.magnitude <= MAX_MAGNITUDE,
            s2.magnitude <= MAX_MAGNITUDE,
            s3.magnitude <= MAX_MAGNITUDE,
            s1.id is not s2.id, s2.id is not s3.id, s1.id is not s3,
            not any(s1.unit_vector.T * s2.unit_vector >= np.cos(CAMERA_FOV)),
            not any(s1.unit_vector.T * s3.unit_vector >= np.cos(CAMERA_FOV)),
            not any(s2.unit_vector.T * s3.unit_vector >= np.cos(CAMERA_FOV)),
        ])


generator = CatalogueGenerator()
generator.generate_triangles()
