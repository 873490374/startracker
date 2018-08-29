import math
from typing import Union

import numpy as np
from numba import jit

from program.const import COS_CAMERA_FOV
from program.planar_triangle import ImagePlanarTriangle, CatalogPlanarTriangle
from program.star import StarUV
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class StarIdentifier:

    def __init__(
            self,
            planar_triangle_calculator: PlanarTriangleCalculator,
            kvector_calculator: KVectorCalculator,
            catalog: np.ndarray):
        self.planar_triangle_calc = planar_triangle_calculator
        self.kvector_calc = kvector_calculator
        self.catalog = catalog

    def identify_stars(
            self, image_stars: [StarUV],
            previous_frame_stars: np.ndarray = None):
        if previous_frame_stars:
            stars = self.identify(image_stars, previous_frame_stars)
        else:
            stars = self.identify(image_stars, self.catalog)
        return stars

    def identify(
            self, star_list: np.ndarray, previous_triangles: np.ndarray
    ) -> Union[np.ndarray, None]:
        i = 0
        unique_found_triangles = []
        for s1 in star_list[:-2]:
            # TODO s1 unique triangles
            i += 1
            j = i
            for s2 in star_list[i:-1]:
                # TODO s2 unique triangles
                j += 1
                k = j
                # previous_triangle = None
                for s3 in star_list[k:]:
                    # TODO s3 unique triangles
                    t = self.planar_triangle_calc.calculate_triangle(
                        s1, s2, s3)
                    ct = self.find_in_catalog(t)
                    if ct.size == 0:
                        ct = self.catalog
                    if len(ct) == 1:
                        if (len(unique_found_triangles) > 0 and
                                (unique_found_triangles == ct[0]).any()):
                            return ct[0]
                        else:
                            unique_found_triangles.append(ct[0])
                    else:
                        res = self.get_two_common_stars_triangles(
                            s1, s2, s3, ct, star_list)
                        if res.size == 0:
                            continue
                        if len(res) == 1:
                            if (len(unique_found_triangles) > 0 and
                                    (unique_found_triangles == res[0]).any()):
                                return res[0]
                            else:
                                unique_found_triangles.append(res[0])
        # print("***")
        # [print(t) for t in unique_found_triangles]
        # print("***")
        return unique_found_triangles

    def find_in_catalog(
            self, triangle: np.ndarray) -> np.ndarray:
        # triangle = [id1, id2, id3, area, moment, area_var, moment_var]
        A_dev = np.math.sqrt(triangle[5])
        J_dev = np.math.sqrt(triangle[6])
        area_min = triangle[3] - A_dev
        area_max = triangle[3] + A_dev
        moment_min = triangle[4] - J_dev
        moment_max = triangle[4] + J_dev

        k_start, k_end = self.kvector_calc.find_in_kvector(
                moment_min, moment_max, self.catalog)
        # TODO make it faster by using numpy arrays or GPU

        valid_triangles = self.catalog[
            (self.catalog[:, 5] >= k_start) &
            (self.catalog[:, 5] <= k_end) &
            (self.catalog[:, 3] >= area_min) &
            (self.catalog[:, 3] <= area_max) &
            (self.catalog[:, 4] >= moment_min) &
            (self.catalog[:, 4] <= moment_max)]

        # if valid_triangles.size == 0:
        #     return self.catalog

        return valid_triangles

    def get_two_common_stars_triangles(
            self, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray,
            ct: np.ndarray, star_list: np.ndarray
    ) -> [CatalogPlanarTriangle]:
        triangles = ct
        for star_couple in [(s1, s2), (s1, s3), (s2, s3)]:
            sc1 = star_couple[0]
            sc2 = star_couple[1]

            for sc3 in star_list:
                # Pivot for each star couple
                # TODO return somehow star id
                if are_the_same_stars(sc1, sc2, sc3):
                    continue

                t = self.planar_triangle_calc.calculate_triangle(sc1, sc2, sc3)
                tc = self.find_in_catalog(t)
                # if tc.size == 0:
                # This is slower
                #     continue
                triangles = self.find_common_triangles(triangles, tc)
                if len(triangles) == 1:
                    return triangles
                if len(triangles) == 0:
                    return triangles
        return triangles

    def find_common_triangles(
            self, previous_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
        # TODO this is slow and called a lot!!!
        # print(len(previous_t), len(new_t))
        C = array_row_intersection(previous_t, new_t)
        # print(len(C))
        return C


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]


def are_the_same_stars(sc1, sc2, sc3):
    return (sc1[0] == sc2[0] or
            sc2[0] == sc3[0] or
            sc3[0] == sc1[0])
