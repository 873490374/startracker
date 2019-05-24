import numpy as np


class OrientationFinder:
    def __init__(self, quest_calculator, star_catalog):
        self.quest_calc = quest_calculator
        self.star_catalog = star_catalog

    def find_orientation(self, current_stars, previous_stars=None):
        if not current_stars:
            return None
        current_stars = self.remove_false_stars(current_stars)
        if not current_stars:
            return None
        if not previous_stars:
            previous_stars = self.find_previous_stars(current_stars)
        previous_vectors, current_vectors = self.sort_vectors(
            np.array(previous_stars), np.array(current_stars)[:, 1:5])
        weight_list = [1 for _ in range(len(current_stars))]
        q, K_calc = self.quest_calc.calculate_quest(
            weight_list, current_vectors, previous_vectors)
        return q

    @staticmethod
    def remove_false_stars(current_stars):
        curr = []
        for s in current_stars:
            if s[1] != -1:
                curr.append(s)
        return curr

    def find_previous_stars(self, current_stars):
        return self.star_catalog[
            np.isin(self.star_catalog[:, 0], np.array(current_stars)[:, 1])]

    @staticmethod
    def sort_vectors(prev_vectors, curr_vectors):
        assert len(prev_vectors) == len(curr_vectors)
        prev_vectors = prev_vectors[prev_vectors[:, 0].argsort()]
        curr_vectors = curr_vectors[curr_vectors[:, 0].argsort()]
        for l in range(len(prev_vectors)):
            assert curr_vectors[l][0] == prev_vectors[l][0]
        return prev_vectors[:, 1:], curr_vectors[:, 1:]
