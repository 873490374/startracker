from collections import Counter
from itertools import combinations

import numpy as np

from program.const import COS_CAMERA_FOV
from program.const import SIG_X
from program.parallel.kvector_calculator_parallel import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class StarIdentifier:

    def __init__(
            self,
            planar_triangle_calculator: PlanarTriangleCalculator,
            kvector_calculator: KVectorCalculator,
            triangle_catalog: np.ndarray,
            star_catalog: np.ndarray):
        self.planar_triangle_calc = planar_triangle_calculator
        self.kvector_calc = kvector_calculator
        self.triangle_catalog = triangle_catalog
        self.star_catalog = star_catalog

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
            except IndexError:
                continue
        try:
            result_stars = []
            ids = found_stars.keys()
            for id in ids:
                try:
                    id_cat = int(Counter(found_stars[id]).most_common(1)[0][0])
                    s = image_stars[id]
                    result_stars.append(
                        np.array([s[0], id_cat, s[1], s[2], s[3]]))
                except (KeyError, IndexError):
                    continue
            result_stars = self.verify_stars(result_stars)
            return result_stars
        except KeyError:
            return None

    def find_stars(self, s1, s2, s3):
        triangle_catalog_stars = []
        image_triangle = self.planar_triangle_calc.calculate_triangle(
            s1, s2, s3)
        catalog_triangles = self.find_in_catalog(image_triangle)
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

        # k_start, k_end = self.kvector_calc.find_in_kvector(
        #     area_min, area_max, self.catalog)
        # TODO should I make it faster with GPU?

        valid_triangles = self.triangle_catalog[
            # TODO why k-vector does not work?
            # (self.catalog[:, 5] >= k_start+440) &
            # (self.catalog[:, 5] <= k_end+480) &
            (self.triangle_catalog[:, 3] >= area_min) &
            (self.triangle_catalog[:, 3] <= area_max) &
            (self.triangle_catalog[:, 4] >= moment_min) &
            (self.triangle_catalog[:, 4] <= moment_max)]

        return valid_triangles

    def verify_stars(self, result_stars: []):
        # [id_img, id_cat, uv0, uv1, uv2]
        # TODO CUDA?
        # TODO verify if wrong stars (useful in case there are false stars)
        result_stars2 = np.array(result_stars)
        result_stars2 = self.star_catalog[
            np.isin(self.star_catalog[:, 0], np.array(result_stars2)[:, 1])]

        wrong_stars = []

        comb = combinations(result_stars2, 2)
        for c in comb:
            s1 = c[0]
            s2 = c[1]
            l1 = s1[1] * s2[1] + s1[2] * s2[2] + s1[3] * s2[3]

            if l1 < COS_CAMERA_FOV:
                wrong_stars.append(s1[0])
                wrong_stars.append(s2[0])

        count = Counter(wrong_stars).most_common(len(result_stars2))
        print('hello')
        # get stars uv from star catalog (by newly found id)
        # combinations
        # check that each star combination with such id can exist on this scene
        # by checking if their angle is inside the FOV
        # in case of some error, say this scene was not identified
        # OR maybe remove wrongly identified stars (the ones giving error)
        return result_stars
