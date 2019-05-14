from collections import Counter
from itertools import combinations

import numpy as np

from program.const import SIG_X
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


class Tracker:

    def __init__(self, planar_triangle_calculator: PlanarTriangleCalculator):
        self.planar_triangle_calc = planar_triangle_calculator

    def track(
            self,
            new_image_stars: np.ndarray,  # [id_img, uv1, uv2, uv3]
            previous_found_stars: np.ndarray,
            # [id_img, id_cat, uv1, uv2, uv3]
    ):
        previous_found_stars2 = [s[1:] for s in previous_found_stars]
        previous_triangles = self.make_triangles_list(previous_found_stars2)
        found_stars = self.identify_stars(new_image_stars, previous_triangles)
        found_stars = self.choose_best_stars(found_stars, new_image_stars)
        return found_stars

    def make_triangles_list(self, image_stars: [np.ndarray]) -> [np.ndarray]:
        found_stars = {key: [] for key in range(len(image_stars))}
        comb = combinations(found_stars, 3)
        triangles = []
        for c in comb:
            # TODO change to CUDA
            s1 = image_stars[c[0]]
            s2 = image_stars[c[1]]
            s3 = image_stars[c[2]]
            triangle = self.planar_triangle_calc.calculate_triangle(
                s1, s2, s3)
            triangles.append(triangle)
        return triangles

    def identify_stars(self, new_image_stars, previous_triangles):
        found_stars = {key: [] for key in range(len(new_image_stars))}
        comb = combinations(found_stars, 3)
        for c in comb:
            # TODO change to CUDA
            s1 = new_image_stars[c[0]]
            s2 = new_image_stars[c[1]]
            s3 = new_image_stars[c[2]]
            stars = self.find_stars(s1, s2, s3, previous_triangles)
            try:
                found_stars[s1[0]].extend(stars)
                found_stars[s2[0]].extend(stars)
                found_stars[s3[0]].extend(stars)
            except IndexError:
                continue
        return found_stars

    def find_stars(self, s1, s2, s3, previous_triangles):
        triangle_catalog_stars = []
        image_triangle = self.planar_triangle_calc.calculate_triangle(
            s1, s2, s3)
        catalog_triangles = self.find_common_triangles(
            image_triangle, previous_triangles)
        triangle_catalog_stars.extend(catalog_triangles[:, 0])
        triangle_catalog_stars.extend(catalog_triangles[:, 1])
        triangle_catalog_stars.extend(catalog_triangles[:, 2])
        return triangle_catalog_stars

    def choose_best_stars(self, found_stars, new_image_stars):
        try:
            result_stars = []
            ids = found_stars.keys()
            result_ids = []
            for id in ids:
                try:
                    most_common = Counter(found_stars[id]).most_common(3)
                    for l in range(3):
                        id_cat = int(most_common[l][0])
                        if id_cat not in result_ids:
                            result_ids.append(id_cat)
                            s = new_image_stars[id]
                            result_stars.append(
                                np.array([s[0], id_cat, s[1], s[2], s[3]]))
                            break
                except (KeyError, IndexError):
                    continue
            return result_stars
        except KeyError:
            return None

    def find_common_triangles(
            self, triangle: np.ndarray, previous_triangles: [np.ndarray]):
        # triangle = [id1, id2, id3, area, moment, area_var, moment_var]
        A_dev = np.math.sqrt(triangle[5])
        J_dev = np.math.sqrt(triangle[6])
        area_min = triangle[3] - SIG_X * A_dev
        area_max = triangle[3] + SIG_X * A_dev
        moment_min = triangle[4] - SIG_X * J_dev
        moment_max = triangle[4] + SIG_X * J_dev

        # TODO should I make it faster with GPU?
        previous_triangles = np.array(previous_triangles)
        valid_triangles = previous_triangles[
            (previous_triangles[:, 3] >= area_min) &
            (previous_triangles[:, 3] <= area_max) &
            (previous_triangles[:, 4] >= moment_min) &
            (previous_triangles[:, 4] <= moment_max)]

        return valid_triangles
