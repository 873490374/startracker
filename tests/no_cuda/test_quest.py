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
        assert np.isclose(quaternion.ra, 54.95945, rtol=1.e-6, atol=1.e-8)
        assert np.isclose(quaternion.dec, 14.89439, rtol=1.e-6, atol=1.e-8)

        assert np.isclose(vi1, np.dot(R, vb1), atol=0.015).all()
        assert np.isclose(vi2, np.dot(R, vb2), atol=0.018).all()

        R_diff = np.inner(R.T, R_bi_quest_expected)
        assert np.isclose(np.identity(3), R_diff, atol=0.004).all()
        assert np.isclose(R.T, R_bi_quest_expected, atol=0.004).all()
        print_ori(wahba(vb_list, vi_list, weight_list))


def wahba(A, B, weight=[]):
    """
    Takes in two matrices of points and finds the attitude matrix needed to
    transform one onto the other
    Input:
        A: nx3 matrix - x,y,z in body frame
        B: nx3 matrix - x,y,z in eci
        Note: the "n" dimension of both matrices must match
    Output:
        attitude_matrix: returned as a numpy matrix
    """
    assert len(A) == len(B)
    if (len(weight) == 0):
        weight = np.array([1] * len(A))
    # dot is matrix multiplication for array
    H = np.dot(np.transpose(A) * weight, B)

    # calculate attitude matrix
    # from http://malcolmdshuster.com/FC_MarkleyMortari_Girdwood_1999_AAS.pdf
    U, S, Vt = np.linalg.svd(H)
    flip = np.linalg.det(U) * np.linalg.det(Vt)

    # S=np.diag([1,1,flip]); U=np.dot(U,S)
    U[:, 2] *= flip

    body2ECI = np.dot(U, Vt)
    return body2ECI


def print_ori(body2ECI):
    # DEC=np.degrees(np.arcsin(body2ECI[2,0]))
    # rotation about the z axis (-180 to +180)
    # RA=np.degrees(np.arctan2(body2ECI[1,0],body2ECI[0,0]))
    # rotation about the camera axis (-180 to +180)
    #  ORIENTATION=np.degrees(-np.arctan2(body2ECI[1,2],body2ECI[2,2]))
    DEC = np.degrees(np.arcsin(body2ECI[0, 2]))
    RA = np.degrees(np.arctan2(body2ECI[0, 1], body2ECI[0, 0]))
    ORIENTATION = np.degrees(-np.arctan2(body2ECI[1, 2], body2ECI[2, 2]))
    if ORIENTATION > 180:
        ORIENTATION = ORIENTATION - 360

    # rotation about the y axis (-90 to +90)
    print("DEC=" + str(DEC))
    # rotation about the z axis (-180 to +180)
    print("RA=" + str(RA))
    # rotation about the camera axis (-180 to +180)
    print("ORIENTATION=" + str(ORIENTATION))
