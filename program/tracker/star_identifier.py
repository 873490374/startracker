from typing import Union

import numpy as np

from program.planar_triangle import ImagePlanarTriangle, CatalogPlanarTriangle
from program.star import StarUV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class StarIdentifier:

    def __init__(
            self, planar_triangle_calculator: PlanarTriangleCalculator,
            sensor_variance: float, max_magnitude: int,
            camera_fov: int, catalog: [CatalogPlanarTriangle]):
        self.planar_triangle_calc = planar_triangle_calculator
        self.max_magnitude = max_magnitude
        self.sensor_variance = sensor_variance
        self.camera_fov = camera_fov
        self.catalog = catalog

    def identify_stars(
            self, image_stars: [StarUV],
            previous_frame_stars: [CatalogPlanarTriangle]=None):
        if previous_frame_stars:
            stars = self.identify(image_stars, previous_frame_stars)
        else:
            stars = self.identify(image_stars, self.catalog)
        return stars

    def find_in_catalog(
            self, triangle: ImagePlanarTriangle) -> [CatalogPlanarTriangle]:

        A_dev = np.math.sqrt(triangle.area_var)  # * 0.03177
        J_dev = np.math.sqrt(triangle.moment_var)  # * 0.03177
        area_min = triangle.area - A_dev
        area_max = triangle.area + A_dev
        moment_min = triangle.moment - J_dev
        moment_max = triangle.moment + J_dev

        valid_triangles = []
        for tt in self.catalog:
            if (area_min <= tt.area <= area_max and
                    moment_min <= tt.moment <= moment_max):
                valid_triangles.append(tt)
        # print('found in catalog', len(valid_triangles))
        # if len(valid_triangles) == 1:
        #     print(valid_triangles[0])
        return valid_triangles

    def get_two_common_stars_triangles(
            self, s1: StarUV, s2: StarUV, s3: StarUV,
            ct: [CatalogPlanarTriangle], star_list: [ImagePlanarTriangle]
    ) -> [CatalogPlanarTriangle]:
        triangles = ct
        for star_couple in [(s1, s2), (s1, s3), (s2, s3)]:
            sc1 = star_couple[0]
            sc2 = star_couple[1]

            for sc3 in star_list:
                if self.are_stars_valid(
                        sc1, sc2, sc3, self.max_magnitude, self.camera_fov):
                    continue
                t = self.planar_triangle_calc.calculate_triangle(sc1, sc2, sc3)
                tc = self.find_in_catalog(t)
                triangles = self.find_common_triangles(triangles, tc)
                if len(triangles) == 1:
                    # print('One triangle found', len(triangles))
                    return triangles
                if len(triangles) == 0:
                    # print('No triangles found', len(triangles))
                    return triangles
        # print('Number of stars after all two common stars triangles',
        #       len(triangles))
        return triangles

    def find_common_triangles(
            self, previous_t: [CatalogPlanarTriangle],
            new_t: [CatalogPlanarTriangle]) -> [CatalogPlanarTriangle]:
        common_triangles = []
        for t1 in previous_t:
            for t2 in new_t:
                if t1.has_the_same_stars(t2):
                    common_triangles.append(t2)
                    # print('common_triangles')
        return common_triangles

    def identify(self, star_list: [StarUV],
                 previous_triangles: [CatalogPlanarTriangle]
                 ) -> Union[CatalogPlanarTriangle, None]:
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
                        # return ct[0]
                        print('one star?', ct[0])
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
