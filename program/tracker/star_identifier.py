import copy

import numpy as np

from progress.bar import Bar

from program.classes import StarUV
from program.tracker.planar_triangle import PlanarTriangle


class Triangle:
    def __init__(self, A: float, A_var: float, J: float, J_var: float, s1: StarUV, s2: StarUV, s3: StarUV):
        self.A = A
        self.A_var = A_var
        self.J = J
        self.J_var = J_var
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class StarIdentifier:

    def __init__(self, sensor_variance: int, max_magnitude: int,
                 camera_fov: int, catalog: np.ndarray):
        self.sensor_variance = sensor_variance
        self.max_magnitude = max_magnitude
        self.camera_fov = camera_fov
        self.catalog = catalog

    def identify_stars(self, stars_list: [StarUV]):
        triangles = self.get_triangles(stars_list)
        star_ids = self.find_in_catalog(triangles)
        return star_ids

    def get_triangles(self, stars_list: [StarUV]) -> [Triangle]:
        triangles = []
        for s1 in stars_list:
            for s2 in stars_list:
                for s3 in stars_list:
                    if is_valid(
                            s1, s2, s3, self.max_magnitude, self.camera_fov):
                        # print(s1.magnitude, s2.magnitude, s3.magnitude)
                        triangle = PlanarTriangle()
                        triangle.calculate_triangle(
                            s1.unit_vector,
                            s2.unit_vector,
                            s3.unit_vector,
                            self.sensor_variance
                        )
                        # print(triangle.A, triangle.A_var,
                        #       triangle.J, triangle.J_var)
                        triangles.append(Triangle(
                            triangle.A, triangle.A_var,
                            triangle.J, triangle.J_var,
                            s1, s2, s3))
        return triangles

    def find_in_catalog(self, triangles: [Triangle]):
        catalog_copy = copy.deepcopy(self.catalog)
        catalog_copy = [t for t in self.catalog]
        print('before', len(catalog_copy))
        for t in triangles:
            area_min = t.A - t.A_var
            area_max = t.A + t.A_var
            moment_min = t.J - t.J_var
            moment_max = t.J + t.J_var
            # print(area_min, area_max, moment_min, moment_max)
            trian_to_delete = []
            for tt in catalog_copy:
                if not (area_min <= tt[3] <= area_max and
                        moment_min <= tt[4] <= moment_max):
                    # print(tt[0], tt[1], tt[2])
                    trian_to_delete.append(tt)
                    # print('deleted')
            for td in trian_to_delete:
                catalog_copy.remove(td)
            print('left: ', len(catalog_copy))
        # print(catalog_copy)
        print('after', len(catalog_copy))


def is_valid(
        s1: StarUV, s2: StarUV, s3: StarUV,
        max_magnitude: int, camera_fov: int) -> bool:
    if any([all(s1.unit_vector == s2.unit_vector),
            all(s1.unit_vector == s3.unit_vector),
            all(s2.unit_vector == s3.unit_vector)]):
        return False
    # print(s1.magnitude, s2.magnitude, s3.magnitude)
    # print(s1.unit_vector, s2.unit_vector, s3.unit_vector)
    # print(s1.unit_vector.T * s2.unit_vector)
    return all([
        s1.magnitude <= max_magnitude,
        s2.magnitude <= max_magnitude,
        s3.magnitude <= max_magnitude,
        not any(s1.unit_vector.T * s2.unit_vector >= np.cos(camera_fov)),
        not any(s1.unit_vector.T * s3.unit_vector >= np.cos(camera_fov)),
        not any(s2.unit_vector.T * s3.unit_vector >= np.cos(camera_fov)),
    ])
