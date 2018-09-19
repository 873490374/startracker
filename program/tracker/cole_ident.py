import math

import numpy as np
from scipy import misc


def CalcFOV(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    
    # Point is also the vector from the origin to the point.
    
    # Vector r from each pt to another pt. Length of each r is a side of a triangle
    
    r12 = p2 - p1
    r23 = p3 - p2
    r31 = p1 - p3
    
    # Calc length of vector from pt to pt
    
    a = math.sqrt( r12[1]**2 + r12[2]**2 + r12[3]**2 )
    b = math.sqrt( r23[1]**2 + r23[2]**2 + r23[3]**2 )
    c = math.sqrt( r31[1]**2 + r31[2]**2 + r31[3]**2 )
    
    # Determine interior angles of triangle
    
    A = math.acos( (b**2 + c**2 - a**2)/(2*b*c) )
    # A*180/pi
    B = math.acos( (a**2 + c**2 - b**2)/(2*a*c) )
    # B*180/pi
    C = math.acos( (a**2 + b**2 - c**2)/(2*b*a) )
    # C*180/pi
    
    # If all the angles within the triangle are less than 90 degrees, the field of
    # view is deterimned by drawing a circle through the three pts
    
    # If any angle is greater than 90 degrees, the field of view is determined
    # by the greatest distance from any pt to another.
    
    r = 0
    
    if max( [ A, B, C ] ) < (math.pi/2):
        
        # CalcFOV determines the field of view occupied by three pts on a sphere
        # p1, p2 and p3 are arrays: [ x y z ]
        
        # Distance to center from pt 1 to pt 2 must be equal (eq 1)
        
        A1 = [- 2*p1[1] + 2*p2[1], - 2*p1[2] + 2*p2[2], - 2*p1[3] + 2*p2[3]]
        B1 = - (p1[1]**2 + p1[2]**2 + p1[3]**2) + (p2[1]**2 + p2[2]**2 + p2[3]**2)
        
        # Distance to center from pt 2 to pt 3 must be equal [eq 2]
        
        A2 = [- 2*p2[1] + 2*p3[1], - 2*p2[2] + 2*p3[2], - 2*p2[3] + 2*p3[3]]
        B2 = - (p2[1]**2 + p2[2]**2 + p2[3]**2) + (p3[1]**2 + p3[2]**2 + p3[3]**2)
        
        # All points must be on the same plane (eq 3)
        
        P = np.array(
            [[np.transpose(-p1)],
             [np.dot(np.transpose(p2), -p1)],
             [np.dot(np.transpose(p3), -p1)]]
        )
        
        A3 = [
            ( P[2,2] * P[3,3] ) - ( P[2,3] * P[3,2] ),
            ( P[2,3] * P[3,1] ) - ( P[2,1] * P[3,3] ),
            ( P[2,1] * P[3,2] ) - ( P[2,2] * P[3,1] ),
        ]
        B3 = - np.linalg.det(P)
        
        B = [B1, B2, B3]
        
        A = [ [A1], [A2], [A3] ]
        
        x = np.dot(np.invert(A), np.transpose(B))
        
        r = math.sqrt( (p1[1]-x[1])**2 + (p1[2]-x[2])**2 + (p1[3]-x[3])**2 )

    else:
        
        r = max( [a, b, c] )/2 # Make negative to indicate "skinny"
        
    
    # FOV is twice the angle formed by triangle with opp=r and hyp=1
    
    alpha = [ 2*math.asin(r/1) ]
    # alpha*180/pi
    return alpha[0]


def identify_stars(StarsInFOV, FOVmax=10, plimit=9, sigx=3, climit=10000):
    nStarsInFOV = len(StarsInFOV)
    if nStarsInFOV < 3:
        return []

    nCombs = misc.comb(nStarsInFOV, 3)

    # CREATE LIST OF TRIANGLES, INCL. AREA, AND MAX & MIN TRIANGLE INDICIES

    k = 0
    start = 1
    maxProp = 0
    T = []

    # Create array of Triangles in FOV, including range of possible
    # solutions from Triangle Catalog using K-Vector

    for s1 in range(1, nStarsInFOV-2):
        sv1 = StarsInFOV(s1).mv
        for s2 in range(s1+1, nStarsInFOV-1):
            sv2 = StarsInFOV(s2).mv

            mAng = math.acos(np.inner(sv1, sv2 ))

            if mAng > math.pi/2:
                mAng = math.pi - mAng

            if mAng <= FOVmax:

                for s3 in range(s2+1, nStarsInFOV):
                    sv3 = StarsInFOV(s3).mv

                    mFOV = CalcFOV(sv1, sv2, sv3)  # Determine if skinny
                    if mFOV <= FOVmax:

                        # Measure area of triangle, determine bounds on
                        # error
                        # TODO calculate triangle
                        Prop = PlanarTriArea( sv1, sv2, sv3 )
                        var  = StarPlanarAreaCov( sv1, sv2, sv3, sigm )

                        # Calculate minimum and maximum areas

                        PropMin = Prop - sigx * math.sqrt(var)
                        PropMax = Prop + sigx * math.sqrt(var)

                        # Determine ranges of possible solutions using K-Vector

                        pmin = FindWithParabKvec(PropMin, Kvec)
                        pmax = FindWithParabKvec(PropMax, Kvec)
                        rangee = pmax - pmin

                        if rangee <= climit:      # Only keep tri list within combo limit
                            # TODO append to T
                            k = k + 1
                            T[k].Stars = [s1, s2, s3]
                            T[k].Prop  = Prop
                            T[k].Prop2 = -1    # Don't fill till needed
                            T[k].pmin  = pmin
                            T[k].pmax  = pmax
                            T[k].llist = 0     # Used for linked list

                            # Find Triangle with greatest area to start list

                            if T[k].Prop > maxProp:
                                start = k
                                maxProp = T[k].Prop

    nCombs = k

    # CREATE LINKED LIST OF TRIANGLES IN FOV, SUCH THAT NO MORE THAN
    # ONE STAR CHANGES AT A TIME GIVING PRIORITY TO TRIANGLES WITH
    # LARGEST POSSIBLE AREAS

    T[start].llist = 999       # makes ineligible (also EOL)
    prev = start
    next = start
    nPivots = 0
    for j in range(1, nCombs):
        maxProp = 0
        done = False

        for k in range (1, nCombs):
            if T[k].llist == 0:
                x = 0
                for m in range (1, 3):
                    if T[k].Stars[m] == T[prev].Stars[1]:
                            x = x + 1
                    if T[k].Stars[m] == T[prev].Stars[2]:
                            x = x + 1
                    if T[k].Stars[m] == T[prev].Stars[3]:
                            x = x + 1

                if x == 2:       # At least two common pts
                    if T[k].Prop > maxProp:
                        next = k
                        maxProp = T[k].Prop

                    done = True

        T[prev].llist = next
        T[next].llist = 999    # EOL
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
        nFinalists = T[start].pmax - T[start].pmin

        k = 0

        # Get Ic of first triangle, find range of allowable Ic

        s1  = T[start].Stars[1]
        s2  = T[start].Stars[2]
        s3  = T[start].Stars[3]
        sv1 = StarsInFOV[s1].mv
        sv2 = StarsInFOV[s2].mv
        sv3 = StarsInFOV[s3].mv
        # T[start].Prop2 = PlanarTriPolarMoment( sv1, sv2, sv3 )
        # var = StarPlanarMomentCov( sv1, sv2, sv3, sigm )
        # Prop2min = T[start].Prop2 - sigx * math.sqrt( var )
        # Prop2max = T[start].Prop2 + sigx * math.sqrt( var )

        for j in range(1, nFinalists):
            tnum = TriPtr[T[start].pmin + j - 1]

            # Include only if Ip is within allowable range
            # TODO read from catalog
            if Tri[tnum].Ip >= Prop2min:
                if Tri[tnum].Ip <= Prop2max:
                    k = k + 1
                    Finalists[k].Tri   = tnum
                    Finalists[k].Stars = Tri[tnum].Stars

        nFinalists = k
        Results.nFinalists = k

    else:
        nFinalists = 0
        Results.nFinalists = 0

    # ==============================================
    # PIVOT AS REQUIRED TO NARROW POSSIBLE SOLUTIONS
    # ==============================================


    fail = 0
    k = T[start].llist

    for j in range(1, nPivots):

        # If number of finalists reduces to 0, search has failed
        # If number of finalists reduces to 1, search is complete


        if nFinalists == 0:
            nPivots = j - 1
            break
        if nFinalists == 1:
            nPivots = j - 1
            break

        # Plot in FOV as desired (bit 2 of gmode set)

        Results.nFinalists = [Results.nFinalists, (T[k].pmax - T[k].pmin)]

        # Create three binary trees of possible triangles,
        # sorted by first star, second start and third star

        # saTree = []
        # sbTree = []
        # scTree = []

        s1 = T[k].Stars[1]
        s2 = T[k].Stars[2]
        s3 = T[k].Stars[3]
        sv1 = StarsInFOV[s1].mv
        sv2 = StarsInFOV[s2].mv
        sv3 = StarsInFOV[s3].mv

        # TODO read from catalog
        triangles = self.find_in_catalog()

        F1 = []    # Reset list of finalists
        n1 = 0

        for a in range(1, nFinalists):
            s1 = Finalists[a].Stars[1, 1]
            s2 = Finalists[a].Stars[1, 2]
            s3 = Finalists[a].Stars[1, 3]

            match = []

            """
            Here is part for finding all two common stars triangles
            """
            # TODO two common stars triangles
            self.common_triangles(s1, s2, s3, tc)

            for b in range(1, len(match)):
                n1 = n1 + 1
                F1[n1].Tri   = [ [match[b]], [Finalists[a].Tri] ]
                F1[n1].Stars = [ [Tri( match[b] ).Stars], [Finalists[a].Stars]]

            if n1 > nFinalists:
                nPivots = j - 1
                break


        # If no of finalists increases from previous round, abandon matching

        if n1 > nFinalists:
            break

        # Newly created list (F1) becomes current finalists

        Finalists = F1
        nFinalists = n1

        # Advance to next combination in linked list

        k = T[k].llist

    # COMPILE RESULTS

    Results.nPivots = nPivots

    if nFinalists == 0:     # Search failed to match triangle
        Results.Match = []
        nResults = 0
    elif nFinalists == 1:   # Search successful, create array of stars in FOV
        Results.Match = Finalists[1].Stars[1]  # .Stars(1, 1:3) all three stars
        n1 = 3
        for j in range(2, nPivots+1):
            for k in range(1, 3):
                match = False
                for m in range(1, n1):
                    if Finalists[1].Stars[j,k] == Results.Match(m):
                        match = True
                        break
                if not match:
                    Results.Match = [ Results.Match, Finalists[1].Stars[j,k] ]
                    n1 = n1 + 1
        nResults = n1
    else:                   # Unable to reduce possible solutions to one
        Results.Match = []
        nResults = 0
    return Results.Match
