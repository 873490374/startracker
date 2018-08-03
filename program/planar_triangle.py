from program.star import StarUV


class CatalogPlanarTriangle:
    def __init__(
            self, s1_id: int, s2_id: int, s3_id: int, area: float,
            moment: float, k: int):
        self.s1_id = s1_id
        self.s2_id = s2_id
        self.s3_id = s3_id
        self.area = area
        self.moment = moment
        self.k = k

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


class ImagePlanarTriangle:
    def __init__(self, s1: StarUV, s2: StarUV, s3: StarUV, area: float,
                 moment: float, area_var: float, moment_var: float,
                 k: int=None):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.area = area
        self.moment = moment
        self.area_var = area_var
        self.moment_var = moment_var
        self.k = k

    def __iter__(self):
        yield 'star1_id', self.s1.id
        yield 'star2_id', self.s2.id,
        yield 'star3_id', self.s3.id,
        yield 'area', self.area
        yield 'moment', self.moment
        yield 'k', self.k

    def __str__(self):
        return "{}, {}, {}, {}, {}, {}".format(
            self.s1, self.s2, self.s3, self.area, self.moment, self.k
        )

    def __eq__(self, other):
        if isinstance(other, ImagePlanarTriangle):
            return all([
                self.s1 == other.s1,
                self.s2 == other.s2,
                self.s3 == other.s3,
                self.area == other.area,
                self.moment == other.moment,
                self.area_var == other.area_var,
                self.moment_var == other.moment_var,
                self.k == other.k,
            ])
        return False
