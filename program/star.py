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

    def __eq__(self, other):
        if isinstance(other, StarPosition):
            return self.id == other.id
        return False


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

    def __eq__(self, other):
        if isinstance(other, StarUV):
            return all([
                np.isclose(self.unit_vector[0], other.unit_vector[0]),
                np.isclose(self.unit_vector[1], other.unit_vector[1]),
                np.isclose(self.unit_vector[2], other.unit_vector[2])])
        return False
