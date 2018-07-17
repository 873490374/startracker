import copy

import numpy as np

from progress.bar import Bar

from program.classes import StarUV
from program.tracker.planar_triangle import PlanarTriangle


class Triangle:
    def __init__(
            self, A: float, A_var: float, J: float, J_var: float,
            s1: StarUV, s2: StarUV, s3: StarUV):
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
        # star_ids = self.find_in_catalog(triangles)
        print(triangles)
        return triangles

    def get_triangles(self, stars_list: [StarUV]) -> [Triangle]:
        triangles = []
        # iterator = iter(stars_list)
        i = 0
        for s1 in stars_list:
            i += 1
            j = 0
            for s2 in stars_list[i:]:
                j += 1
                k = j
                # for s3 in stars_list[j:]:
                for l in range(k):
                    s3 = stars_list[k]
                    print(s3)
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
                        t = Triangle(
                            triangle.A, triangle.A_var,
                            triangle.J, triangle.J_var,
                            s1, s2, s3)
                        found_triangles = self.find_in_catalog(t)
                        if len(found_triangles) == 1:
                            return found_triangles[0]
                        else:
                            try:
                                s4 = stars_list[k+1]
                                # print(s1, s2, s3, s4)
                            except IndexError:
                                # iterator = iter(stars_list)
                                break
                            if is_valid(
                                    s1, s2, s4,
                                    self.max_magnitude, self.camera_fov):
                                triangle2 = PlanarTriangle()
                                triangle2.calculate_triangle(
                                    s1.unit_vector,
                                    s2.unit_vector,
                                    s4.unit_vector,
                                    self.sensor_variance
                                )
                                t2 = Triangle(
                                    triangle2.A, triangle2.A_var,
                                    triangle2.J, triangle2.J_var,
                                    s1, s2, s4)
                                found_triangles2 = self.find_in_catalog(t2)
                                ft = [value for value in
                                      found_triangles if value in
                                      found_triangles2]
                                print(len(ft))
                                if len(ft) == 1:
                                    return ft[0]
                                else:
                                    continue
                        # print(triangle.A, triangle.A_var,
                        #       triangle.J, triangle.J_var)
                        # triangles.append()
        return triangles

    def find_in_catalog(self, triangle: Triangle) -> [Triangle]:
        # catalog_copy = [t for t in self.catalog]
        # print('before', len(catalog_copy))
        # for t in triangles:
        A_dev = np.math.sqrt(triangle.A_var)
        J_dev = np.math.sqrt(triangle.J_var)
        area_min = triangle.A - A_dev
        area_max = triangle.A + A_dev
        moment_min = triangle.J - J_dev
        moment_max = triangle.J + J_dev

        valid_triangles = []
        for tt in self.catalog:
            if (area_min <= tt[3] <= area_max and
                    moment_min <= tt[4] <= moment_max):
                # print(tt[0], tt[1], tt[2])
                # valid_triangles.append(tt)
                valid_triangles.append((tt[0], tt[1], tt[2], tt[3], tt[4]))
        return valid_triangles

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
