import numpy as np

from program.tracker.star_identifier import two_common_stars_triangles


class TestStarIdentifier:

    def test_common_triangles(self):
        tri = np.array([5, 6, 7, 231, 515])
        tc = np.array([
            [1, 2, 3, 222, 232],
            [2, 4, 7, 222, 232],
            [1, 4, 7, 222, 232],
            [6, 7, 9, 222, 232],
            [3, 5, 6, 222, 232],
            [3, 6, 7, 222, 232],
        ])
        result = two_common_stars_triangles(tri, tc)
        expected = np.array([
            [6, 7, 9, 222, 232],
            [3, 5, 6, 222, 232],
            [3, 6, 7, 222, 232],
        ])
        assert (expected == result).all()
