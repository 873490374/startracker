import numpy as np

b = np.matrix('1 2 3; 4 5 6; 7 8 9')
# print(a)
# print(b)

a = b - 1
# print(a)
# print(type(a))
# print(a.A)

i = 0
for x in a.A:
    j = 0
    for y in a.A.T:
        print(a[i, j])
        j += 1
    i += 1

i_bottom = sum()
i_top = 0
i_left = 0
i_right = 0
