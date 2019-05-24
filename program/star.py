
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
