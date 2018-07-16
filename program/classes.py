import numpy as np


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

    def __iter__(self):
        yield 'star1_id', self.star1_id
        yield 'star2_id', self.star2_id,
        yield 'star3_id', self.star3_id,
        yield 'area', self.area
        yield 'polar_moment', self.polar_moment
