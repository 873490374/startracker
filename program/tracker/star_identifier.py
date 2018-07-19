from typing import Union

import numpy as np

from program.planar_triangle import PlanarTriangleImage, PlanarTriangleCatalog
from program.star import StarUV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class StarIdentifier:

    def __init__(self,
                 planar_triangle_calculator: PlanarTriangleCalculator,
                 sensor_variance: int, max_magnitude: int,
                 camera_fov: int, catalog: [PlanarTriangleCatalog]):
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

    def find_in_catalog(
            self, triangle: PlanarTriangleImage) -> [PlanarTriangleCatalog]:

        A_dev = np.math.sqrt(triangle.area_var)  # * 0.03177
        J_dev = np.math.sqrt(triangle.moment_var)  # * 0.03177
        area_min = triangle.area - A_dev
        area_max = triangle.area + A_dev
        moment_min = triangle.moment - J_dev
        moment_max = triangle.moment + J_dev

        valid_triangles = []
        for tt in self.catalog:
            if (area_min <= tt[3] <= area_max and
                    moment_min <= tt[4] <= moment_max):
                # print(tt[0], tt[1], tt[2])
                valid_triangles.append(tt)
                # valid_triangles.append((int(tt[0]), int(tt[1]), int(tt[2]), tt[3], tt[4]))
        return valid_triangles

    def get_two_common_stars_triangles(
            self, s1: StarUV, s2: StarUV, s3: StarUV,
            ct: [PlanarTriangleCatalog], star_list: [PlanarTriangleImage]) -> \
            [PlanarTriangleCatalog]:
        for star_couple in [(s1, s2), (s1, s3), (s2, s3)]:
            sc1 = star_couple[0]
            sc2 = star_couple[1]
            # new ct???
            for sc3 in star_list:
                if self.are_stars_valid(
                        sc1, sc2, sc3, self.max_magnitude, self.camera_fov):
                    continue
                t = self.planar_triangle_calc.calculate_triangle(sc1, sc2, sc3)
                tc = self.find_in_catalog(t)
                ct = self.find_common_triangles(ct, tc)
                if len(ct) == 1:
                    print('One triangle found', len(ct))
                    return ct[0]
                if len(ct) == 0:
                    print('No triangles found', len(ct))
                    return None
        print('Number of stars after all two common stars triangles', len(ct))
        return None

    def find_common_triangles(
            self, previous_t: [PlanarTriangleCatalog],
            new_t: [PlanarTriangleCatalog]) -> [PlanarTriangleCatalog]:
        common_triangles = []
        for t1 in previous_t:
            for t2 in new_t:
                if t1.has_the_same_stars(t2):
                    common_triangles.append(t2)
        return common_triangles

    def identify(self, star_list: [StarUV],
                 previous_triangles: [PlanarTriangleCatalog]
                 ) -> Union[PlanarTriangleCatalog, None]:
        i = 0
        for s1 in star_list[:-2]:
            i += 1
            j = 0
            for s2 in star_list[i:-1]:
                j += 1
                k = j
                # previous_triangle = None
                for s3 in star_list[k:]:
                    if not self.are_stars_valid(
                            s1, s2, s3, self.max_magnitude, self.camera_fov):
                        continue
                    t = self.planar_triangle_calc.calculate_triangle(
                        s1, s2, s3)
                    ct = self.find_in_catalog(t)
                    if len(ct) == 1:
                        return ct[0]
                    else:
                        res = self.get_two_common_stars_triangles(
                            s1, s2, s3, ct, star_list)
                        if len(res) == 1:
                            return res[0]
        return None

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
