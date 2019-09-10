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
            np.array([ 0.00000000e+00,  3.08830000e+04, -7.03935481e-02,  2.27089231e-02,
                    9.97260775e-01,  5.67124711e+02,  8.69346107e+01, -1.55818806e-01,
       -6.37934299e-02,  9.85723540e-01]),
            np.array([ 1.00000000e+00, -1.00000000e+00, -4.23232348e-02,  7.67300727e-02,
        9.96153221e-01,  8.46186979e+02,  2.31468804e+02, -1.28884241e-01,
       -1.03744604e-02,  9.91605377e-01]),
            np.array([ 2.00000000e+00, -1.00000000e+00,  9.90270988e-04, -1.08321553e-02,
        9.99940840e-01,  3.94281258e+02,  4.55093783e+02, -8.57660560e-02,
       -9.74890895e-02,  9.91534196e-01]),
            np.array([ 3.00000000e+00,  3.22460000e+04, -4.99458311e-02, -7.99376389e-02,
        9.95547783e-01,  3.70001076e+01,  1.91953564e+02, -1.34526147e-01,
       -1.63966718e-01,  9.77250035e-01]),
            np.array([ 4.00000000e+00,  3.03430000e+04, -2.48609265e-02,  1.19371244e-02,
        9.99619647e-01,  5.11422242e+02,  3.22078584e+02, -1.11343119e-01,
       -7.48639143e-02,  9.90958175e-01]),
            np.array([ 5.00000000e+00,  2.96960000e+04,  8.17443363e-02, -5.39578538e-02,
        9.95191647e-01,  1.71125582e+02,  8.72485377e+02, -5.29638360e-03,
       -1.40303521e-01,  9.90094374e-01]),
            np.array([ 6.00000000e+00,  2.96550000e+04, -1.65145115e-03,  3.48906912e-02,
        9.99389770e-01,  6.29570671e+02,  4.41500536e+02, -8.86675532e-02,
       -5.22973500e-02,  9.94687414e-01]),
            np.array([ 7.00000000e+00,  2.87340000e+04,  3.86956736e-02,  5.50958909e-02,
        9.97730970e-01,  7.34031487e+02,  6.49484745e+02, -4.86220768e-02,
       -3.22125443e-02,  9.98297674e-01])]
        attitude_finder.find_attitude(current_stars)
