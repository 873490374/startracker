import math
from typing import Union

import numpy as np

from program.planar_triangle import ImagePlanarTriangle, CatalogPlanarTriangle
from program.star import StarUV
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class StarIdentifier:

    def __init__(
            self, planar_triangle_calculator: PlanarTriangleCalculator,
            kvector_calculator: KVectorCalculator, max_magnitude: int,
            camera_fov: int, catalog: [CatalogPlanarTriangle]):
        self.planar_triangle_calc = planar_triangle_calculator
        self.kvector_calc = kvector_calculator
        self.max_magnitude = max_magnitude
        self.camera_fov = camera_fov
        self.catalog = catalog

    def identify_stars(
            self, image_stars: [StarUV],
            previous_frame_stars: [CatalogPlanarTriangle] = None):
        if previous_frame_stars:
            stars = self.identify(image_stars, previous_frame_stars)
        else:
            stars = self.identify(image_stars, self.catalog)
        return stars

    def find_in_catalog(
            self, triangle: ImagePlanarTriangle) -> [CatalogPlanarTriangle]:

        A_dev = np.math.sqrt(triangle.area_var)
        J_dev = np.math.sqrt(triangle.moment_var)
        area_min = triangle.area - A_dev
        area_max = triangle.area + A_dev
        moment_min = triangle.moment - J_dev
        moment_max = triangle.moment + J_dev

        if any([
            triangle.area == 0,
            triangle.moment == 0,
            area_min is math.nan,
            area_max is math.nan,
            moment_min is math.nan,
            moment_max is math.nan,
        ]):
            return self.catalog

        smaller_catalog = self.kvector_calc.find_in_kvector(
            moment_min, moment_max, self.catalog)

        valid_triangles = []
        # for tt in self.catalog:
        for tt in smaller_catalog:
            if (area_min <= tt.area <= area_max and
                    moment_min <= tt.moment <= moment_max):
                valid_triangles.append(tt)
        # print('found in catalog', len(valid_triangles))
        # if len(valid_triangles) == 1:
        #     print(valid_triangles[0])

        if len(valid_triangles) == 0:
            return self.catalog
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
                    return triangles
                if len(triangles) == 0:
                    return triangles
        return triangles

    def find_common_triangles(
            self, previous_t: [CatalogPlanarTriangle],
            new_t: [CatalogPlanarTriangle]) -> [CatalogPlanarTriangle]:
        common_triangles = []
        for t1 in previous_t:
            for t2 in new_t:
                if t1.has_the_same_stars(t2):
                    common_triangles.append(t2)
        return common_triangles

    def identify(self, star_list: [StarUV],
                 previous_triangles: [CatalogPlanarTriangle]
                 ) -> Union[CatalogPlanarTriangle, None]:
        i = 0
        unique_found_triangles = []
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
                        # if ct[0] in unique_found_triangles:
                            return ct[0]
                        # else:
                        #     unique_found_triangles.append(ct[0])
                    else:
                        res = self.get_two_common_stars_triangles(
                            s1, s2, s3, ct, star_list)
                        if len(res) == 1:
                            # if res[0] in unique_found_triangles:
                                return res[0]
                            # else:
                            #     unique_found_triangles.append(res[0])
        # print("***")
        # [print(t) for t in unique_found_triangles]
        # print("***")
        return None

    def are_stars_valid(self, s1: StarUV, s2: StarUV, s3: StarUV,
                        max_magnitude: float, camera_fov: int) -> bool:
        if any([s1 == s2, s1 == s3, s2 == s3]):
            return False
        # if not any([s1.id in scene_ids and s2.id in scene_ids,
        #             s1.id in scene_ids and s3.id in scene_ids,
        #             s2.id in scene_ids and s3.id in scene_ids]):
        #     return False
        #
        # if any([
        #     s1.id not in scene_ids,
        #     s2.id not in scene_ids,
        #     s3.id not in scene_ids]):
        #     return False
        if not all([
            s1.magnitude <= max_magnitude,
            s2.magnitude <= max_magnitude,
            s3.magnitude <= max_magnitude,
        ]):
            return False
        return all([
            np.inner(s1.unit_vector.T, s2.unit_vector
                     ) >= np.cos(np.deg2rad(camera_fov)),
            np.inner(s2.unit_vector.T, s3.unit_vector
                     ) >= np.cos(np.deg2rad(camera_fov)),
            np.inner(s1.unit_vector.T, s3.unit_vector
                     ) >= np.cos(np.deg2rad(camera_fov)),
        ])


scene_ids = [
    # 78401, 80112, 78820,
    # 35904, 33579, 34444,
    # 71681, 71683, 68702,
    # 85927, 85696, 86228, 87073, 86670,
    # 41037, 44382, 45101, 41312, 45080, 43783,
    # 85696, 86228, 87073, 86670, 85927,
    # 78820, 80763, 78265, 78401, 81266, 80112
]
