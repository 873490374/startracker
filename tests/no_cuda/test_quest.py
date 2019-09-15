import numpy as np
from Quaternion import Quat

from program.tracker.quest import QuestCalculator


# noinspection PyUnusedLocal
from program.utils import vector_to_angles


class TestQuest:

    def test_quest(self):
        vi1 = np.array([  # inertial frame / catalog
            0.2673,
            0.5345,
            0.8018
        ])
        vi2 = np.array([
            -0.3124,
            0.9370,
            0.1562
        ])

        vb1_exact = np.array([  # body frame / real measured vectors
            0.7749,
            0.3448,
            0.5297
        ])
        vb2_exact = np.array([
            0.6296,
            0.6944,
            -0.3486
        ])

        vb1 = np.array([  # body frame / measured vectors
            0.7814,
            0.3751,
            0.4987
        ])
        vb2 = np.array([
            0.6164,
            0.7075,
            -0.3459
        ])

        weight_list = [1, 1]
        vb_list = np.array([vb1, vb2])
        vi_list = np.array([vi1, vi2])

        R_bi_exact = np.array([
            [0.5335, 0.8080, 0.2500],
            [-0.8080, 0.3995, 0.4330],
            [0.2500, -0.4330, 0.8660]
        ])

        R_bi_quest_expected = np.array([
            [0.5571, 0.7895, 0.2575],
            [-0.7950, 0.4175, 0.4400],
            [0.2399, -0.4499, 0.8603]
        ])

        eigenvalue_expected = 1.9996

        eigenvector_expected = np.array([0.2643, -0.0051, 0.4706, 0.8418])
        q_expected = np.array(
            [0.26430369, -0.00510007, 0.47060657, 0.84181174])

        K_expected = np.array([
            [-1.1929, 0.8744, 0.9641, 0.4688],
            [0.8744, 0.5013, 0.3536, -0.4815],
            [0.9641, 0.3536, -0.5340, 1.1159],
            [0.4688, -0.4815, 1.1159, 1.2256],
        ])

        phi_expected = 1.773  # attitude error in degrees
        J_expected = 3.6810e-4  # loss function value

        quest_calc = QuestCalculator()
        q, K_calc = quest_calc.calculate_quest(weight_list, vb_list, vi_list)
        assert np.isclose(np.array(q), q_expected, atol=0.002).all()
        assert np.isclose(K_calc, K_expected, atol=0.001).all()

        quaternion = Quat(q)
        R = quaternion.transform

        assert np.isclose(vi1, np.dot(R, vb1), atol=0.015).all()
        assert np.isclose(vi2, np.dot(R, vb2), atol=0.018).all()

        R_diff = np.inner(R.T, R_bi_quest_expected)
        assert np.isclose(np.identity(3), R_diff, atol=0.004).all()
        assert np.isclose(R.T, R_bi_quest_expected, atol=0.004).all()

        vi1a = vector_to_angles(vi1)
        vi2a = vector_to_angles(vi2)
        vb1a = vector_to_angles(np.dot(R, vb1))
        vb2a = vector_to_angles(np.dot(R, vb2))
        vb1exa = vector_to_angles(np.dot(R, vb1_exact))
        vb2exa = vector_to_angles(np.dot(R, vb2_exact))
        assert np.isclose(vi1a, vb1a, atol=2).all()
        assert np.isclose(vi2a, vb2a, atol=2).all()
        assert np.isclose(vi1a, vb1exa, atol=3).all()
        assert np.isclose(vi2a, vb2exa, atol=2).all()
