import os

import numpy as np

from program.const import MAIN_PATH
from program.tracker.attitude_finder import AttitudeFinder
from program.tracker.quest import QuestCalculator


class TestAttitudeFinder:

    def test_attitude_finder(self):
        filename_star = os.path.join(
            MAIN_PATH, 'tests/catalog/star_catalog_mag6.2.csv')
        with open(filename_star, 'rb') as f:
            star_catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        attitude_finder = AttitudeFinder(
            quest_calculator=QuestCalculator(),
            star_catalog=star_catalog,
        )
        current_stars = [
            np.array([0.00000000e+00, 4.25700000e+04, 2.98497352e-02,
                      9.41754077e-02, 9.95108027e-01]),
            np.array([1.00000000e+00,  3.85000000e+04,  1.24566609e-02,
                      -5.23422130e-02, 9.98551513e-01]),
            np.array([2.00000000e+00,  3.80890000e+04, -3.37250524e-02,
                      -4.80272739e-02, 9.98276516e-01]),
            np.array([3.00000000e+00,  3.85180000e+04, -1.15912377e-02,
                      -4.13939758e-02, 9.99075664e-01]),
            np.array([4.00000000e+00,  4.00960000e+04, -6.95179081e-02,
                      4.60352812e-02, 9.96517944e-01]),
            np.array([5.00000000e+00, 4.10390000e+04, 3.11490386e-02,
                      3.21810779e-02, 9.98996554e-01]),
            np.array([6.00000000e+00,  3.81640000e+04, -4.40955306e-02,
                      -4.08790135e-02, 9.98190608e-01]),
            np.array([7.00000000e+00,  3.99530000e+04, -4.11659732e-03,
                      7.57042918e-03, 9.99962870e-01]),
            np.array([8.00000000e+00,  3.89570000e+04,  1.23680668e-02,
                      -3.62269417e-02, 9.99267051e-01]),
            np.array([9.00000000e+00, 4.26240000e+04, 4.07151763e-02,
                      8.95958706e-02, 9.95145645e-01])]
        attitude_finder.find_attitude(current_stars)
