import numpy as np
from PIL import Image

# 1.
a_roi = 5
i_threshold = 100


def image_to_matrix(img):
    return np.asarray(img.convert('L'))

image = Image.open("2.jpg")
I = image_to_matrix(image)
print(I)


def sum_border(start, end, where, constant, matrix):
    m = matrix
    i = start
    border_sum = 0
    while i < end:
        if where is 'rows':
            print(I[i, constant])
            border_sum += I[i, constant]
        elif where is 'columns':
            border_sum += I[constant, i]
        i += 1
    return border_sum

blabla = 0
x = 0
for k in I:
    y = 0
    for l in I:
        if I[x, y] > i_threshold:
            x_start = x - (a_roi - 1)/2
            y_start = y - (a_roi - 1)/2
            x_end = x_start + a_roi
            y_end = y_start + a_roi
    # 2.
            if x_start < 0 or y_start < 0:
                y += 1
                continue
    # 3.
            #print(x, y)
            #print(x_start, y_start)
            i_bottom = sum_border(x_start, x_end-1, 'row', y_start, I)
            #print(i_bottom)
            i_top = sum_border(x_start+1, x_end, 'row', y_end, I)
            #print(i_top)
            i_left = sum_border(y_start, y_end-1, 'column', x_start, I)
            #print(i_left)
            i_right = sum_border(y_start+1, y_end, 'column', x_end, I)
            #print(i_right)
            i_border = (i_top + i_bottom + i_left + i_right) / 4 * (a_roi - 1)
            #print(i_border)
            blabla += 1
            continue
    # 4. - normalization
            I_norm_matrix = np.matrix([])
            i = x_start
            while i <= x_end:
                j = y_start
                while j <= y_end:
                    I_norm_matrix = I - i_border

            # 5.

            b = 0

            x_cm = 0

            y_cm = 0

            # 6.

            # 7.

            u = 0

            y += 1

    x += 1
print(blabla)