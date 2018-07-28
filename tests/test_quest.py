import numpy as np

from program.tracker.quest import QuestCalculator


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


        R_bi_quest = np.array([
            [0.5571, 0.7895, 0.2575],
            [-0.7950, 0.4175, 0.4400],
            [0.2399, -0.4499, 0.8603]
        ])
        sigma = 1.773
        J = 3.6810 * np.math.pow(10, -4)
        quest_calc = QuestCalculator()
        attitude = quest_calc.calculate_quest(weight_list, v_list, w_list)
        print(attitude)
