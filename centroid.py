__author__ = 'Szymon Michalski'

import numpy as np
from PIL import Image

# 1.
a_roi = 5
i_threshold = 5


def image_to_matrix(img):
    return np.asarray(img.convert('L'))

img = Image.open("2.jpg")
I = image_to_matrix(img)
print(I)
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
            print(x, y)
            print(x_start, y_start)
            i_bottom = sum()
            i_top = 0
            i_left = 0
            i_right = 0
            i_border = (i_top + i_bottom + i_left + i_right) / 4 * (a_roi - 1)

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
