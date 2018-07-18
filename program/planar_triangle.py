
class PlanarTriangleCatalog:
    def __init__(
            self, s1_id, s2_id, s3_id, area, moment, area_var, moment_var):
        self.s1_id = s1_id
        self.s2_id = s2_id
        self.s3_id = s3_id
        self.area = area
        self.moment = moment
        self.area_var = area_var
        self.moment_var = moment_var

    def has_the_same_stars(self, other):
        this_stars = [self.s1_id, self.s2_id, self.s3_id]
        return all([
            other.s1_id in this_stars,
            other.s2_id in this_stars,
            other.s3_id in this_stars])


class PlanarTriangleImage:
    def __init__(self, area, moment, area_var, moment_var):

        self.area = area
        self.moment = moment
        self.area_var = area_var
        self.moment_var = moment_var

    # def i
