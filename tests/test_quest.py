import numpy as np

from startracker.quest import QuestCalculator


class TestQuest:

    def test_quest(self):
        w1 = np.array([
            0.2673,
            0.5345,
            0.8018
        ])[np.newaxis].T
        w2 = np.array([
            -0.3124,
            0.9370,
            0.1562
        ])[np.newaxis].T

        v1_exact = np.array([
            0.7749,
            0.3448,
            0.5297
        ])[np.newaxis].T
        v2_exact = np.array([
            0.6296,
            0.6944,
            -0.3486
        ])[np.newaxis].T

        v1 = np.array([
            0.7814,
            0.3751,
            0.4987
        ])[np.newaxis].T
        v2 = np.array([
            0.6164,
            0.7075,
            -0.3459
        ])[np.newaxis].T

        weight_list = [1, 1]
        v_list = [v1, v2]
        w_list = [w1, w2]

        quest_calc = QuestCalculator()
        quest_calc.calculate_quest(weight_list, v_list, w_list)
        #
        # p = np.matrix([1.5, 2.4, 1.3])
        # q = np.matrix([14.5, 1.6, 55.1])
        # r = np.matrix([16.5, 112.4, 63.5])
        #
        # planar_triangle = PlanarTriangle()
        # planar_triangle.calculate_triangle(p, q, r)
        #
        # print(planar_triangle.A)
        # print(planar_triangle.A_var)
        # print(planar_triangle.J)
        # print(planar_triangle.J_var)
