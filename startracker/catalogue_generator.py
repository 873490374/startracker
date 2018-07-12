import numpy as np


MAX_MAGNITUDE = 999
CAMERA_FOV = 999


class StarPosition:
    def __init__(self, star_id, magnitude, right_ascension, declination):
        self.id = star_id
        self.magnitude = magnitude
        self.right_ascension = right_ascension
        self.declination = declination


class StarUV:
    def __init__(self, star_id, magnitude, unit_vector):
        self.id = star_id
        self.magnitude = magnitude
        self.unit_vector = unit_vector


class CatalogueTriangle:
    def __init__(self, star1_id, star2_id, star3_id, area, polar_moment):
        self.star1_id = star1_id
        self.star2_id = star2_id
        self.star3_id = star3_id
        self.area = area
        self.polar_moment = polar_moment


class CatalogueGenerator:

    def generate(self, observatory_catalogue):
        return []

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

    def is_valid(self, s1: StarUV, s2: StarUV) -> bool:
        return (
            s1.magnitude <= MAX_MAGNITUDE and
            s2.magnitude <= MAX_MAGNITUDE and
            s1.unit_vector.T*s2.unit_vector >= np.cos(CAMERA_FOV)
        ).all()
