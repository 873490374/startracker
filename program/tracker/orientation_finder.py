import numpy as np


class OrientationFinder:
    def __init__(self, quest_calculator, star_catalog):
        self.quest_calc = quest_calculator
        self.star_catalog = star_catalog

    def find_orientation(self, current_stars, previous_stars=None):
        if not previous_stars:
            previous_stars = self.find_previous_stars(current_stars)
        assert len(current_stars) == len(previous_stars)
        current_vectors = self.sort_vectors(current_stars)
        previous_vectors = self.sort_vectors(previous_stars)
        weight_list = [1 * len(current_stars)]
        q, K_calc = self.quest_calc.calculate_quest(
            weight_list, current_vectors, previous_vectors)
        return q

    def find_previous_stars(self, current_stars):
        return self.star_catalog[
            np.isin(self.star_catalog[:, 0], np.array(current_stars)[:, 1])]

    def sort_vectors(self, vectors):
        vectors = vectors[vectors[:, 0].argsort()]
        return vectors[1:]
