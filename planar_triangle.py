import math
import numpy as np

p = np.array([0, 0])
q = np.array([1, 0])
r = np.array([0, 1])

a = np.linalg.norm(p - q)
b = np.linalg.norm(q - r)
c = np.linalg.norm(p - r)
"""
It's the magnitude of a vector. For example, if you have v = <a, b> = ai + bj, then
||v|| = √(a^2 + b^2), and if v = <a, b, c> = ai + bj + ck, then v = √(a^2 + b^2 + c^2).

So if you have C(t) = x(t)i + y(t)j, then C'(t) = x'(t)i + y'(t)j, and ||C'(t)|| = √{[x'(t)]^2 + [y'(t)]^2}.
"""
s = 0.5 * (a + b + c)

A = math.sqrt(s * (s - a) * (s - b) * (s - c))

J = A * (a ** 2 + b ** 2 + c ** 2) / 36

print(J)

# Variance - Area

u1 = (s - a) * (s - b) * (s - c)
u2 = s * (s - b) * (s - c)
u3 = s * (s - a) * (s - c)
u4 = s * (s - a) * (s - b)

dA_da = (u1 - u2 + u3 + u4) / 4 * A
dA_db = (u1 + u2 - u3 + u4) / 4 * A
dA_dc = (u1 + u2 + u3 - u4) / 4 * A

da_db1 = (p - q).T / a
db_db2 = (q - r).T / b
dc_db1 = (p - r).T / c
da_db2 = - da_db1
db_db3 = - db_db2
dc_db3 = - dc_db1

h1T = dA_da * da_db1 + dA_dc * dc_db1
h2T = dA_da * da_db2 + dA_db * db_db2
h3T = dA_db * db_db3 + dA_dc * dc_db3

H = [h1T, h2T, h3T]

R1 = 
R2 =
R3 =

R = np.matrix([R1, 0, 0, 0, 0, 0, 0],
              [],
              [])

variance_area = H * R * H.T


# Variance - Polar Moment

dJ_da = A * a / 18
dJ_db = A * b / 18
dJ_dc = A * c / 18
dJ_dA = (a**2 + b**2 + c**2) / 36

h1T = dJ_da * da_db1 + dJ_dc * dc_db1 + dJ_dA * h1T
h2T = dJ_da * da_db2 + dJ_db * db_db2 + dJ_dA * h2T
h3T = dJ_db * db_db3 + dJ_dc * dc_db3 + dJ_dA * h3T

H = [h1T, h2T, h3T]

R1 =
R2 =
R3 =

R = np.matrix([R1, 0, 0, 0, 0, 0, 0],
              [],
              [])

variance_moment = H * R * H.T
