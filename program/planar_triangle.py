from program.star import StarUV


class PlanarTriangleCatalog:
    def __init__(
            self, s1_id: int, s2_id: int, s3_id: int, area: float,
            moment: float):
        self.s1_id = s1_id
        self.s2_id = s2_id
        self.s3_id = s3_id
        self.area = area
        self.moment = moment

    def __str__(self):
        return "{}, {}, {}".format(
            self.s1_id, self.s2_id, self.s3_id
        )

    def has_the_same_stars(self, other):
        this_stars = [self.s1_id, self.s2_id, self.s3_id]
        return all([
            other.s1_id in this_stars,
            other.s2_id in this_stars,
            other.s3_id in this_stars])


class PlanarTriangleImage:
    def __init__(self, s1: StarUV, s2: StarUV, s3: StarUV, area: float,
                 moment: float, area_var: float, moment_var: float):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.area = area
        self.moment = moment
        self.area_var = area_var
        self.moment_var = moment_var
