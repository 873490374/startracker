import itertools
import numpy as np
import time
from PIL import Image

start_time = time.time()
# 1.
pixel_size = 5
focal_length = 7
a_roi = 5
i_threshold = 100


def image_to_matrix(img):
    return np.asarray(img.convert('L'))


def sum_border(start, end, where, constant, matrix):
    m = matrix
    i = start
    border_sum = 0
    while i < end:
        if where == 'row':
            border_sum += m[i, constant]
        elif where == 'column':
            border_sum += m[constant, i]
        i += 1
    return border_sum


def total_mass(x_start, x_end, y_start, y_end, I_norm):
    mass_sum = 0
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            mass_sum += I_norm[x, y]
            y += 1
        x += 1
    return mass_sum


def array_point_mass(x_start, x_end, y_start, y_end, I_norm, b):
    x_cm = 0
    y_cm = 0
    x = x_start
    while x < x_end:
        y = y_start
        while y < y_end:
            x_cm = I_norm[x, y] * x / b
            y_cm = I_norm[x, y] * y / b
            y += 1
        x += 1
    return x_cm, y_cm


image = Image.open("2.jpg")
I = image_to_matrix(image)
# print(I)

star_list = []
x = 0

for k in I:
    y = 0
    for l in I:
        if I[x, y] > i_threshold:
            x_start = int(x - (a_roi - 1)/2)
            y_start = int(y - (a_roi - 1)/2)
            x_end = int(x_start + a_roi)
            y_end = int(y_start + a_roi)
    # 2.
            #print(len(I), len(I.T))
            if x_start < 0 or y_start < 0 or x_end > len(I)-1 or y_end > len(I.T)-1:
                y += 1
                continue
    # 3.
            # print(x, y)
            # print(x_start, x_end, y_start, y_end)
            i_bottom = sum_border(x_start, x_end-1, 'row', y_start, I)
            i_top = sum_border(x_start+1, x_end, 'row', y_end, I)
            i_left = sum_border(y_start, y_end-1, 'column', x_start, I)
            i_right = sum_border(y_start+1, y_end, 'column', x_end, I)
            i_border = (i_top + i_bottom + i_left + i_right) / 4 * (a_roi - 1)

    # 4. - normalization
            I_norm_matrix = np.matrix([])

            I_norm_matrix = I - i_border

            # print(I_norm_matrix)

    # 5.

            b = total_mass(x_start+1, x_end-1, y_start+1, y_end-1, I_norm_matrix)
            # print(b)

            x_cm, y_cm = array_point_mass(x_start+1, x_end-1, y_start+1, y_end-1, I_norm_matrix, b)

            #print(x_cm, y_cm)
            star_list.append([x_cm, y_cm])
        y += 1
    x += 1

#print(star_list)

# 6. Clustering
control = 1
pixel_diff = 1
while control > 0:
    star_list2 = star_list
    control = 0
    for star1, star2 in itertools.combinations(star_list2, 2):
        if (star2[0] + pixel_diff >= star1[0] >= star2[0] - pixel_diff or
                star2[1] + pixel_diff >= star1[1] >= star2[1] - pixel_diff):
            control += 1

            star_list.remove(star1)
            star_list.remove(star2)
            star_list.append([(star1[0] + star2[0]) / 2, (star1[1] + star2[1]) / 2])
            break


vectors_list = []

# 7. unit vector u
for star in star_list:
    vector = np.array([pixel_size * star[0], pixel_size * star[1], focal_length])
    u = vector.T / np.linalg.norm(vector)
    vectors_list.append(u)

end_time = time.time()

# validation

print(len(star_list))

print(end_time - start_time)
print(star_list)

import png

img = np.zeros((len(I), len(I.T)), dtype='float16')
for star in star_list:
    img[int(star[0]), int(star[1])] = 100
    print(img[int(star[0]), int(star[1])])
    print(img[int(star[0]), int(star[1])-1])
png.from_array(img, 'L').save('test.png')
