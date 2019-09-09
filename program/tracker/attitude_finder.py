import numpy as np
from Quaternion import Quat


class AttitudeFinder:
    def __init__(self, quest_calculator, star_catalog):
        self.quest_calc = quest_calculator
        self.star_catalog = star_catalog

    def find_attitude(self, current_stars, previous_stars=None):
        if not current_stars:
            return None
        current_stars = self.remove_false_stars(current_stars)
        if not current_stars:
            return None
        if not previous_stars:
            previous_stars = self.find_previous_stars(current_stars)
        current_stars = np.delete(current_stars, 6, 1)
        current_stars = np.delete(current_stars, 5, 1)
        current_stars = np.delete(current_stars, 4, 1)
        current_stars = np.delete(current_stars, 3, 1)
        current_stars = np.delete(current_stars, 2, 1)
        current_stars = np.delete(current_stars, 0, 1)
        previous_vectors, current_vectors = self.sort_vectors(
            np.array(previous_stars), np.array(current_stars))
        weight_list = [1 for _ in range(len(current_stars))]
        print_ori(wahba(current_vectors, previous_vectors, weight_list))
        q, K_calc = self.quest_calc.calculate_quest(
            weight_list, current_vectors, previous_vectors)
        q = Quat(q)
        print(q.equatorial)
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


def wahba(A, B, weight=None):
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
    if weight is None:
        weight = []
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
    ##rotation about the z axis (-180 to +180)
    # RA=np.degrees(np.arctan2(body2ECI[1,0],body2ECI[0,0]))
    ##rotation about the camera axis (-180 to +180)
    # ORIENTATION=np.degrees(-np.arctan2(body2ECI[1,2],body2ECI[2,2]))
    DEC = np.degrees(np.arcsin(body2ECI[0, 2]))
    RA = np.degrees(np.arctan2(body2ECI[0, 1], body2ECI[0, 0]))
    ORIENTATION = np.degrees(-np.arctan2(body2ECI[1, 2], body2ECI[2, 2]))
    if ORIENTATION > 180:
        ORIENTATION = ORIENTATION - 360

    # rotation about the z axis (-180 to +180)
    # print("RA=" + str(RA))
    # rotation about the y axis (-90 to +90)
    # print("DEC=" + str(DEC))
    # rotation about the camera axis (-180 to +180)
    # print("ORIENTATION=" + str(ORIENTATION))
    print('DCM:')
    print([RA, DEC, ORIENTATION])
