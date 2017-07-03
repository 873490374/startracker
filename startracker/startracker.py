import math
import numpy

__author__ = 'Szymon Michalski'

vec = [0, 1, 2, 3, 4]


def sum_squares(vec):
    res = 0
    for elem in vec:
        res += elem ^ 2
    return res


def vector_length(vec):
    vec_len = math.sqrt(sum_squares(vec))
    print(vec_len)
    return vec_len

vector_length(vec)
a = numpy.array([1, 3])
b = numpy.array([9, 5])
x = a + b
print(x)

