from collections import Counter
from itertools import combinations

import numpy as np

from program.const import CAMERA_FOV
from program.const import SIG_X
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class StarIdentifier:

    def __init__(
            self,
            planar_triangle_calculator: PlanarTriangleCalculator,
            triangle_catalog: np.ndarray,
            star_catalog: np.ndarray):
        self.planar_triangle_calc = planar_triangle_calculator
        self.triangle_catalog = triangle_catalog
        self.star_catalog = star_catalog
        self.fov = np.cos(np.deg2rad(CAMERA_FOV + 2))

    def identify_stars(self, image_stars: np.ndarray) -> ():
        # [[id_img, uv0, uv1, uv2], ...]
        if len(image_stars) < 3:
            return None
        found_stars = {key: [] for key in range(len(image_stars))}
        comb = combinations(found_stars, 3)
        for c in comb:
            # TODO change to CUDA
            s1 = image_stars[c[0]]
            s2 = image_stars[c[1]]
            s3 = image_stars[c[2]]
            stars = self.find_stars(s1, s2, s3)
            try:
                found_stars[s1[0]].extend(stars)
                found_stars[s2[0]].extend(stars)
                found_stars[s3[0]].extend(stars)
            except (IndexError, TypeError):
                continue
        try:
            result_ids = []
            result_stars = []
            ids = found_stars.keys()
            for id_ in ids:
                try:
                    most_common_limit = 10
                    most_common = Counter(found_stars[id_]).most_common(
                        most_common_limit)
                    if not most_common:
                        s = image_stars[id_]
                        result_stars.append(
                            np.array([s[0], -1, s[1], s[2], s[3]]))
                        continue
                    for i in range(most_common_limit):
                        id_cat = int(most_common[i][0])
                        if id_cat not in result_ids:
                            result_ids.append(id_cat)
                            s = image_stars[id_]
                            result_stars.append(
                                np.array([s[0], id_cat, s[1], s[2], s[3]]))
                            break
                        if i == most_common_limit and id_cat in result_ids:
                            s = image_stars[id_]
                            result_stars.append(
                                np.array([s[0], -1, s[1], s[2], s[3]]))
                except (KeyError, IndexError):
                    continue
            return self.remove_incorrect_stars(result_stars)
        except KeyError:
            return None

    def find_stars(self, s1, s2, s3):
        triangle_catalog_stars = []
        image_triangle = self.planar_triangle_calc.calculate_triangle(
            s1, s2, s3)
        catalog_triangles = self.find_in_catalog(image_triangle)
        if catalog_triangles.size == 0:
            return
        triangle_catalog_stars.extend(catalog_triangles[:, 0])
        triangle_catalog_stars.extend(catalog_triangles[:, 1])
        triangle_catalog_stars.extend(catalog_triangles[:, 2])
        return triangle_catalog_stars

    def find_in_catalog(
            self, triangle: np.ndarray) -> np.ndarray:
        # triangle = [id1, id2, id3, area, moment, area_var, moment_var]
        A_dev = np.math.sqrt(triangle[5])
        J_dev = np.math.sqrt(triangle[6])
        area_min = triangle[3] - SIG_X * A_dev
        area_max = triangle[3] + SIG_X * A_dev
        moment_min = triangle[4] - SIG_X * J_dev
        moment_max = triangle[4] + SIG_X * J_dev

        # TODO change to CUDA

        valid_triangles = self.triangle_catalog[
            (self.triangle_catalog[:, 3] >= area_min) &
            (self.triangle_catalog[:, 3] <= area_max) &
            (self.triangle_catalog[:, 4] >= moment_min) &
            (self.triangle_catalog[:, 4] <= moment_max)]

        return valid_triangles

    def remove_incorrect_stars(self, result_stars: []):
        # [id_img, id_cat, uv0, uv1, uv2]
        result_stars2 = np.array(result_stars)
        result_stars2 = self.star_catalog[
            np.isin(self.star_catalog[:, 0], np.array(result_stars2)[:, 1])]

        wrong_stars = []

        comb = combinations(result_stars2, 2)
        for c in comb:
            s1 = c[0]
            s2 = c[1]
            l1 = s1[1] * s2[1] + s1[2] * s2[2] + s1[3] * s2[3]

            if l1 < self.fov:
                wrong_stars.append(s1[0])
                wrong_stars.append(s2[0])

        wrong_stars = Counter(wrong_stars).most_common(len(result_stars2))
        result_stars = mark_wrong_stars(result_stars, wrong_stars)

        return result_stars


def mark_wrong_stars(found_stars, wrong_stars):
    for s in wrong_stars:
        if s[1] >= 0.5 * len(wrong_stars):
            for star in found_stars:
                if star[1] == s[0]:
                    star[1] = -1
    return found_stars
