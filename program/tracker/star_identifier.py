from collections import Counter

import numpy as np

from program.const import SIG_X
from program.planar_triangle import CatalogPlanarTriangle
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

    def identify_stars(self, image_stars: np.ndarray) -> ():
        s1_id = 0
        unique_found_triangles = []
        found_stars = dict()
        for s1 in image_stars[:-2]:
            s2_id = s1_id + 1
            for s2 in image_stars[s2_id:-1]:
                s3_id = s2_id + 1
                # TODO here calculate for each star couple
                # (another function - inside loop- GPU maybe?)
                # TODO some calculation limit inside maybe?
                try:
                    res1, res2 = self.find_triangles(s1, s2, image_stars)
                    try:
                        found_stars[s1[0]].extend([res1, res2])
                    except KeyError:
                        found_stars[s1[0]] = [res1, res2]
                    try:
                        found_stars[s2[0]].extend([res1, res2])
                    except KeyError:
                        found_stars[s2[0]] = [res1, res2]
                except IndexError:
                    continue
                s2_id += 1
            s1_id += 1
        try:
            ids = found_stars.keys()
            ids = [int(k) for k in ids]
            id1 = ids[0]
            id2 = ids[1]
            id3 = ids[2]
            i1 = Counter(found_stars[id1]).most_common(1)[0][0]
            i2 = Counter(found_stars[id2]).most_common(1)[0][0]
            i3 = Counter(found_stars[id3]).most_common(1)[0][0]
            return (
                image_stars[id1], image_stars[id2], image_stars[id3],
                np.array([i1, i2, i3]))
        except KeyError:
            return None

    def find_triangles(self, s1, s2, image_stars):
        result_triangles = []
        s3_id = int(s2[0] + 1)
        for s3 in image_stars[s3_id:]:
            image_triangle = self.planar_triangle_calc.calculate_triangle(
                s1, s2, s3)
            catalog_triangles = self.find_in_catalog(image_triangle)
            result_triangles.extend(catalog_triangles[:, 0])
            result_triangles.extend(catalog_triangles[:, 1])
            result_triangles.extend(catalog_triangles[:, 2])
        res = Counter(result_triangles).most_common(2)
        return res[0][0], res[1][0]

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

        valid_triangles = self.catalog[
            # TODO why k-vector does not work?
            # (self.catalog[:, 5] >= k_start) &
            # (self.catalog[:, 5] <= k_end) &
            (self.catalog[:, 3] >= area_min) &
            (self.catalog[:, 3] <= area_max) &
            (self.catalog[:, 4] >= moment_min) &
            (self.catalog[:, 4] <= moment_max)]

        # if valid_triangles.size == 0:
        #     return self.catalog
        # valid_triangles = np.delete(valid_triangles, [3, 4, 5], axis=1)
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
                if tc.size == 0:
                    continue
                # TODO correct finding common triangles
                # (two stars, not whole triangle)
                triangles = array_row_intersection(triangles, tc)
                if len(triangles) == 1:
                    return triangles
                if len(triangles) == 0:
                    return triangles
        return triangles


def array_row_intersection(a, b):
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]


def are_the_same_stars(sc1, sc2, sc3):
    return (sc1[0] == sc2[0] or
            sc2[0] == sc3[0] or
            sc3[0] == sc1[0])


def two_common_stars_triangles(tri, tc):
    s1_id = tri[0]
    s2_id = tri[1]
    s3_id = tri[2]

    return tc[
        ((tc[:, 0] == s1_id) & (tc[:, 1] == s2_id)) |
        ((tc[:, 0] == s1_id) & (tc[:, 2] == s2_id)) |

        ((tc[:, 0] == s1_id) & (tc[:, 1] == s3_id)) |
        ((tc[:, 0] == s1_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 1] == s1_id) & (tc[:, 0] == s2_id)) |
        ((tc[:, 1] == s1_id) & (tc[:, 2] == s2_id)) |

        ((tc[:, 1] == s1_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 1] == s1_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 2] == s1_id) & (tc[:, 0] == s2_id)) |
        ((tc[:, 2] == s1_id) & (tc[:, 1] == s2_id)) |

        ((tc[:, 2] == s1_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 2] == s1_id) & (tc[:, 1] == s3_id)) |

        ((tc[:, 0] == s2_id) & (tc[:, 1] == s3_id)) |
        ((tc[:, 0] == s2_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 1] == s2_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 1] == s2_id) & (tc[:, 2] == s3_id)) |

        ((tc[:, 2] == s2_id) & (tc[:, 0] == s3_id)) |
        ((tc[:, 2] == s2_id) & (tc[:, 1] == s3_id))
    ]
