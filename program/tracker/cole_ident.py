import math

import numpy as np
from scipy import misc

from program.parallel.kvector_calculator_parallel import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator


def CalcFOV(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    
    # Point is also the vector from the origin to the point.
    
    # Vector r from each pt to another pt.
    # Length of each r is a side of a triangle
    
    r12 = p2 - p1
    r23 = p3 - p2
    r31 = p1 - p3
    
    # Calc length of vector from pt to pt
    
    a = math.sqrt(r12[0]**2 + r12[1]**2 + r12[2]**2)
    b = math.sqrt(r23[0]**2 + r23[1]**2 + r23[2]**2)
    c = math.sqrt(r31[0]**2 + r31[1]**2 + r31[2]**2)
    
    # Determine interior angles of triangle
    
    A = math.acos((b**2 + c**2 - a**2)/(2*b*c))
    # A*180/pi
    B = math.acos((a**2 + c**2 - b**2)/(2*a*c))
    # B*180/pi
    C = math.acos((a**2 + b**2 - c**2)/(2*b*a))
    # C*180/pi
    
    # If all the angles within the triangle are less than 90 degrees,
    # the field of view is deterimned by drawing a circle through the three pts
    
    # If any angle is greater than 90 degrees, the field of view is determined
    # by the greatest distance from any pt to another.
    
    r = 0
    
    if max([A, B, C]) < (math.pi/2):
        
        # CalcFOV determines the field of view occupied by
        # three pts on a sphere p1, p2 and p3 are arrays: [ x y z ]
        
        # Distance to center from pt 1 to pt 2 must be equal (eq 1)
        
        A1 = [- 2*p1[0] + 2*p2[0], - 2*p1[1] + 2*p2[1], - 2*p1[2] + 2*p2[2]]
        B1 = (- (p1[0]**2 + p1[1]**2 + p1[2]**2) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2))
        
        # Distance to center from pt 2 to pt 3 must be equal [eq 2]
        
        A2 = [- 2*p2[0] + 2*p3[0], - 2*p2[1] + 2*p3[1], - 2*p2[2] + 2*p3[2]]
        B2 = (- (p2[0]**2 + p2[1]**2 + p2[2]**2) +
              (p3[0]**2 + p3[1]**2 + p3[2]**2))
        
        # All points must be on the same plane (eq 3)
        
        P = np.array([np.transpose(-p1),
                      np.transpose(p2) - p1,
                      np.transpose(p3) - p1])
        
        A3 = [
            (P[1, 1] * P[2, 2]) - (P[1, 2] * P[2, 1]),
            (P[1, 2] * P[2, 0]) - (P[1, 0] * P[2, 2]),
            (P[1, 0] * P[2, 1]) - (P[1, 1] * P[2, 0]),
        ]
        B3 = - np.linalg.det(P)
        
        B = np.array([B1, B2, B3])
        
        A = np.array([A1, A2, A3])
        
        x = np.dot(np.linalg.inv(A), np.transpose(B))
        
        r = math.sqrt((p1[0]-x[0])**2 + (p1[1]-x[1])**2 + (p1[2]-x[2])**2)

    else:
        
        r = max([a, b, c])/2  # Make negative to indicate "skinny"
        
    
    # FOV is twice the angle formed by triangle with opp=r and hyp=1
    
    alpha = [2*math.asin(r/1)]
    # alpha*180/pi
    return alpha[0]


FOV_MAX = 10
plimit = 9
SIG_X = 3
climit = 100000


class ColeStarIdentifier:
    def __init__(
            self,
            planar_triangle_calculator: PlanarTriangleCalculator,
            kvector_calculator: KVectorCalculator,
            catalog: np.ndarray):
        self.planar_triangle_calc = planar_triangle_calculator
        self.kvector_calc = kvector_calculator
        self.catalog = catalog

    def identify_stars(self, StarsInFOV):
        nStarsInFOV = len(StarsInFOV)
        if nStarsInFOV < 3:
            return []

        nCombs = misc.comb(nStarsInFOV, 3)

        # CREATE LIST OF TRIANGLES, INCL. AREA, AND MAX & MIN TRIANGLE INDICIES

        k = 0
        start = 1
        max_area = 0
        T = []

        # Create array of Triangles in FOV, including range of possible
        # solutions from Triangle Catalog using K-Vector

        for s1 in range(0, nStarsInFOV-2):
            for s2 in range(s1+1, nStarsInFOV-1):
                # mAng = math.acos(np.inner(sv1, sv2))

                # if mAng > math.pi/2:
                #     mAng = math.pi - mAng

                if 0.1 <= FOV_MAX:

                    for s3 in range(s2+1, nStarsInFOV):

                        # mFOV = CalcFOV(sv1, sv2, sv3)  # Determine if skinny
                        if 0.1 <= FOV_MAX:

                            # Measure area of triangle, determine bounds on
                            # error
                            t = self.planar_triangle_calc.calculate_triangle(
                                StarsInFOV[s1], StarsInFOV[s2], StarsInFOV[s3])

                            area = t[3]
                            moment = t[4]
                            area_var = t[5]
                            moment_var = t[6]
                            llist = 0

                            # TODO K-vector min and max

                            A_dev = np.math.sqrt(area_var)
                            area_min = area - SIG_X * A_dev
                            area_max = area + SIG_X * A_dev

                            k_start, k_end = self.kvector_calc.find_in_kvector(
                                area_min, area_max, self.catalog)
                            rangee = k_end - k_start
                            # Only keep tri list within combo limit
                            if rangee <= climit:
                                T.append([
                                    s1, s2, s3, area, moment,
                                    area_var, moment_var, k_start, k_end, llist
                                ])
                                if area > max_area:
                                    start = k
                                    max_area = area
                                k = k + 1

        nCombs = k

        # CREATE LINKED LIST OF TRIANGLES IN FOV, SUCH THAT NO MORE THAN
        # ONE STAR CHANGES AT A TIME GIVING PRIORITY TO TRIANGLES WITH
        # LARGEST POSSIBLE AREAS

        T[start][9] = 999       # makes ineligible (also EOL)
        prev = start
        next = start
        nPivots = 0
        for j in range(0, nCombs):
            maxProp = 0
            done = False

            for k in range(0, nCombs):
                if T[k][9] == 0:
                    x = 0
                    for m in range(0, 3):
                        if T[k][m] == T[prev][0]:
                                x = x + 1
                        if T[k][m] == T[prev][1]:
                                x = x + 1
                        if T[k][m] == T[prev][2]:
                                x = x + 1

                    if x == 2:       # At least two common pts
                        if T[k][3] > maxProp:
                            next = k
                            maxProp = T[k][3]

                        done = True

            T[prev][9] = next
            T[next][9] = 999    # EOL
            prev = next

            if done:
                nPivots += + 1
                if nPivots == plimit:
                    break
            else:
                break

        # nPivots = nPivots

        # Start with triangle with highest property value

        Finalists = []
        if nCombs > 0:
            nFinalists = T[start][8] - T[start][7]

            # k = 0

            # Get Ic of first triangle, find range of allowable Ic
            #
            # s1 = T[start][0]
            # s2 = T[start][1]
            # s3 = T[start][2]

            # T[start].Prop2 = PlanarTriPolarMoment( sv1, sv2, sv3 )
            # var = StarPlanarMomentCov( sv1, sv2, sv3, sigm )
            # Prop2min = T[start].Prop2 - sigx * math.sqrt( var )
            # Prop2max = T[start].Prop2 + sigx * math.sqrt( var )

            for j in range(0, nFinalists):
                # tnum = TriPtr[T[start][7] + j - 1]

                tf = self.find_in_catalog(np.array(T[start]))
                # Include only if Ip is within allowable range
                # if Tri[tnum].Ip >= Prop2min:
                #     if Tri[tnum].Ip <= Prop2max:
                # k = k + 1
                Finalists.append(tf[:, 0:3])
                # Finalists[k].Tri = tnum
                # Finalists[k].Stars = Tri[tnum].Stars

            nFinalists = len(Finalists)
            RnFinalists = len(Finalists)

        else:
            nFinalists = 0
            RnFinalists = 0

        # ==============================================
        # PIVOT AS REQUIRED TO NARROW POSSIBLE SOLUTIONS
        # ==============================================

        RnFinalists2 = [RnFinalists]
        fail = 0
        k = T[start][9]

        for j in range(0, nPivots):
            # print(nFinalists)
            # If number of finalists reduces to 0, search has failed
            # If number of finalists reduces to 1, search is complete

            if nFinalists == 0:
                nPivots = j - 1
                continue
            if nFinalists == 1:
                nPivots = j - 1
                break

            RnFinalists2.append(T[k][8] - T[k][7])

            # s1 = T[k][0]
            # s2 = T[k][1]
            # s3 = T[k][2]

            triangles = self.find_in_catalog(np.array(T[k]))

            F1 = []    # Reset list of finalists
            n1 = 0

            for a in range(0, nFinalists):
                try:
                    s1 = Finalists[a][0][0]
                    s2 = Finalists[a][0][1]
                    s3 = Finalists[a][0][2]
                except IndexError:
                    nPivots = j - 1
                    break

                match = self.common_triangles(s1, s2, s3, triangles)

                n1 = len(match)
                # for b in range(0, len(match)):
                #     n1 = n1 + 1
                    # F1[n1].Tri = [
                    #     [match[b]],
                    #     [Finalists[a].Tri]]
                F1.append([match[:, 0:3], [Finalists[a][0:3]]])

                if n1 > nFinalists:
                    nPivots = j - 1
                    break

            # If number of finalists increases from previous round,
            # abandon matching

            if n1 > nFinalists:
                break

            # Newly created list (F1) becomes current finalists

            Finalists = F1
            nFinalists = n1

            # Advance to next combination in linked list

            k = T[k][9]

        # COMPILE RESULTS

        RnPivots = nPivots
        # print(nFinalists)
        try:
            ll = [1, 2, 3, match[0, 0:3]]
        except:
            ll = None
        return ll

        RnFinalists = RnFinalists2
        if nFinalists == 0:     # Search failed to match triangle
            RMatch = []
            nResults = 0
        elif nFinalists == 1:
            # Search successful, create array of stars in FOV
            RMatch = Finalists[0]  # .Stars(1, 1:3) all three stars
            n1 = 3
            for j in range(0, nPivots+1):
                for k in range(0, 3):
                    match = False
                    for m in range(0, n1):
                        if Finalists[0][j, k] == RMatch[m]:
                            match = True
                            break
                    if not match:
                        RMatch = [RMatch, Finalists[1][j, k]]
                        n1 = n1 + 1
            nResults = n1
        else:                   # Unable to reduce possible solutions to one
            RMatch = []
            nResults = 0
        return RMatch

    def find_in_catalog(
            self, triangle: np.ndarray) -> np.ndarray:
        # triangle = [id1, id2, id3, area, moment, area_var, moment_var]
        A_dev = np.math.sqrt(triangle[5])
        J_dev = np.math.sqrt(triangle[6])
        area_min = triangle[3] - SIG_X * A_dev
        area_max = triangle[3] + SIG_X * A_dev
        moment_min = triangle[4] - SIG_X * J_dev
        moment_max = triangle[4] + SIG_X * J_dev

        k_start, k_end = self.kvector_calc.find_in_kvector(
                moment_min, moment_max, self.catalog)
        # TODO should I make it faster with GPU?

        valid_triangles = self.catalog[
            # (self.catalog[:, 5] >= k_start) &
            # (self.catalog[:, 5] <= k_end) &
            (self.catalog[:, 3] >= area_min) &
            (self.catalog[:, 3] <= area_max) &
            (self.catalog[:, 4] >= moment_min) &
            (self.catalog[:, 4] <= moment_max)]

        # if valid_triangles.size == 0:
        #     return self.catalog
        # valid_triangles = np.delete(valid_triangles, [3, 4, 5], axis=1)

        return valid_triangles

    def common_triangles(self, s1_id, s2_id, s3_id, tc):

        return tc[
            ((tc[:, 0] == s1_id) & (tc[:, 1] == s2_id)) |
            ((tc[:, 0] == s1_id) & (tc[:, 2] == s2_id)) |

            ((tc[:, 0] == s1_id) & (tc[:, 1] == s3_id)) |
            ((tc[:, 0] == s1_id) & (tc[:, 2] == s3_id)) |

            ((tc[:, 1] == s1_id) & (tc[:, 0] == s2_id)) |
            ((tc[:, 1] == s1_id) & (tc[:, 2] == s2_id)) |

            ((tc[:, 1] == s1_id) & (tc[:, 0] == s3_id)) |
            ((tc[:, 1] == s1_id) & (tc[:, 2] == s3_id)) |

            ((tc[:, 2] == s1_id) & (tc[:, 0] == s2_id)) |
            ((tc[:, 2] == s1_id) & (tc[:, 1] == s2_id)) |

            ((tc[:, 2] == s1_id) & (tc[:, 0] == s3_id)) |
            ((tc[:, 2] == s1_id) & (tc[:, 1] == s3_id)) |

            ((tc[:, 0] == s2_id) & (tc[:, 1] == s3_id)) |
            ((tc[:, 0] == s2_id) & (tc[:, 2] == s3_id)) |

            ((tc[:, 1] == s2_id) & (tc[:, 0] == s3_id)) |
            ((tc[:, 1] == s2_id) & (tc[:, 2] == s3_id)) |

            ((tc[:, 2] == s2_id) & (tc[:, 0] == s3_id)) |
            ((tc[:, 2] == s2_id) & (tc[:, 1] == s3_id))
            ]
