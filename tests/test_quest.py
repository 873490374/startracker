import numpy as np
from Quaternion import Quat

from program.tracker.quest import QuestCalculator


# noinspection PyUnusedLocal
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

        v1_exact = np.array([  # body frame / real measured vectors
            0.7749,
            0.3448,
            0.5297
        ])[np.newaxis].T
        v2_exact = np.array([
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

        assert np.isclose(q, np.array(
            [0.89139439, 0.18244972, -0.17532932, 0.37601565])
        ).all()
        assert np.isclose(K_calc, K_expected, atol=0.001).all()

        quat = Quat([q[1], q[2], q[3], q[0]])
        R = quat.transform
        assert np.isclose(quat.ra, 42.76015, rtol=1.e-6, atol=1.e-8)
        assert np.isclose(quat.dec, 26.72976, rtol=1.e-6, atol=1.e-8)

        assert np.isclose(vi1, np.inner(R, vb1.T), atol=0.12).all()
        assert np.isclose(vi2, np.inner(R, vb2.T), atol=0.26).all()

        R_diff = np.inner(R.T, R_bi_quest_expected)
        assert np.isclose(np.identity(3), R_diff, atol=0.34).all()
        assert np.isclose(
            np.abs(R), np.abs(R_bi_quest_expected), atol=0.26).all()
