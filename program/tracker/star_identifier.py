import numpy as np

from program.planar_triangle import PlanarTriangleImage, PlanarTriangleCatalog
from program.star import StarUV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator

chuj = [42913, 45941, 45556, 41037]


class StarIdentifier:

    def __init__(self,
                 planar_triangle_calculator: PlanarTriangleCalculator,
                 sensor_variance: int, max_magnitude: int,
                 camera_fov: int, catalog: np.ndarray):
        self.planar_triangle_calc = planar_triangle_calculator
        self.sensor_variance = sensor_variance
        self.max_magnitude = max_magnitude
        self.camera_fov = camera_fov
        self.catalog = catalog

    def identify_stars(
            self,
            image_stars: [StarUV],
            previous_frame_stars: [PlanarTriangleCatalog]=None):
        if previous_frame_stars:
            stars = self.identify(image_stars, previous_frame_stars)
        else:
            stars = self.identify(image_stars, self.catalog)
        return stars

    def identify(self, star_list: [StarUV],
                 previous_triangles: [PlanarTriangleCatalog]):
        i = 0
        for s1 in star_list[:-2]:
            i += 1
            j = 0
            for s2 in star_list[i:-1]:
                j += 1
                k = j
                previous_triangle = None
                for s3 in star_list[k:]:
                    t = self.planar_triangle_calc.calculate_triangle(
                        s1, s2, s3)
                    if

        pass

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
                    # print(s3)
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
                        # if [
                        #     int(ggg[0]) in chuj and
                        #     int(ggg[1]) in chuj and
                        #     int(ggg[2]) in chuj for ggg in found_triangles]:
                        #     print('found')
                        #     # print(ggg[0], ggg[1], ggg[2])
                        if len(found_triangles) == 1:
                            return found_triangles[0]
                        else:
                            for s4 in stars_list[k+1:]:

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
                                    print('ft1 ', len(found_triangles))
                                    print('ft2 ', len(found_triangles2))
                                    ft = [value for value in
                                          found_triangles if value in
                                          found_triangles2]
                                    print('common ', len(ft))
                                    print(ft)
                                    if len(ft) == 1:
                                        print('found one')
                                        return ft[0]

                        # print(triangle.A, triangle.A_var,
                        #       triangle.J, triangle.J_var)
                        # triangles.append()
        return triangles

    def find_in_catalog(self, triangle: Triangle) -> [Triangle]:
        # catalog_copy = [t for t in self.catalog]
        # print('before', len(catalog_copy))
        # for t in triangles:
        A_dev = np.math.sqrt(triangle.area_var) * 0.03177
        J_dev = np.math.sqrt(triangle.moment_var) * 0.03177
        area_min = triangle.area - A_dev
        area_max = triangle.area + A_dev
        moment_min = triangle.moment - J_dev
        moment_max = triangle.moment + J_dev

        valid_triangles = []
        for tt in self.catalog:
            if (area_min <= tt[3] <= area_max and
                    moment_min <= tt[4] <= moment_max):
                # print(tt[0], tt[1], tt[2])
                # valid_triangles.append(tt)
                valid_triangles.append((int(tt[0]), int(tt[1]), int(tt[2]), tt[3], tt[4]))
        return valid_triangles

    def convert_star_vectors_to_star_triangles(
            self, star_vectors: [StarUV]) -> [PlanarTriangleImage]:

        stars_list = star_vectors
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
                    # print(s3)
                    if self.are_stars_valid(
                            s1, s2, s3, self.max_magnitude, self.camera_fov):
                        # print(s1.magnitude, s2.magnitude, s3.magnitude)
                        triangle = PlanarTriangle()
                        self.triangle.calculate_triangle(
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
                        # if [
                        #     int(ggg[0]) in chuj and
                        #     int(ggg[1]) in chuj and
                        #     int(ggg[2]) in chuj for ggg in found_triangles]:
                        #     print('found')
                        #     # print(ggg[0], ggg[1], ggg[2])
                        if len(found_triangles) == 1:
                            return found_triangles[0]
                        else:
                            for s4 in stars_list[k + 1:]:

                                if self.are_stars_valid(
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
                                    print('ft1 ', len(found_triangles))
                                    print('ft2 ', len(found_triangles2))
                                    ft = [value for value in
                                          found_triangles if value in
                                          found_triangles2]
                                    print('common ', len(ft))
                                    print(ft)
                                    if len(ft) == 1:
                                        print('found one')
                                        return ft[0]

                        # print(triangle.A, triangle.A_var,
                        #       triangle.J, triangle.J_var)
                        # triangles.append()
        return triangles

    def are_stars_valid(self, s1: StarUV, s2: StarUV, s3: StarUV,
        max_magnitude: float, camera_fov: int) -> bool:
        if any([all(s1.unit_vector == s2.unit_vector),
                all(s1.unit_vector == s3.unit_vector),
                all(s2.unit_vector == s3.unit_vector)]):
            return False
        return all([
            s1.magnitude <= max_magnitude,
            s2.magnitude <= max_magnitude,
            s3.magnitude <= max_magnitude,
            not any(s1.unit_vector.T * s2.unit_vector >= np.cos(camera_fov)),
            not any(s1.unit_vector.T * s3.unit_vector >= np.cos(camera_fov)),
            not any(s2.unit_vector.T * s3.unit_vector >= np.cos(camera_fov)),
        ])
