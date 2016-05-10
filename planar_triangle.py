import math
import numpy as np

__author__ = 'Szymon Michalski'
p = np.array([0, 0])
q = np.array([1, 0])
r = np.array([0, 1])

a = np.linalg.norm(p - q)
b = np.linalg.norm(q - r)
c = np.linalg.norm(p - r)

s = 0.5 * (a + b + c)

A = math.sqrt(s*(s-a)*(s-b)*(s-c))

J = A * (a*a + b*b + c*c)/36

print(J)
