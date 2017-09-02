import itertools
import numpy as np
from PIL import Image


class CentroidCalculator:
    def __init__(self, pixel_size, focal_length, a_roi, i_threshold):
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.a_roi = a_roi
        self.i_threshold = i_threshold

    @staticmethod
    def image_to_matrix(img):
        return np.asarray(img.convert('L'))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def array_point_mass(x_start, x_end, y_start, y_end, I_norm, b):
        x_cm = 0
        y_cm = 0
        x = x_start
        while x < x_end:
            y = y_start
            while y < y_end:
                x_cm += I_norm[x, y] * x / b
                y_cm += I_norm[x, y] * y / b
                y += 1
            x += 1
        return x_cm, y_cm

    def calculate_centroids(self, image):
        I = self.image_to_matrix(image)

        star_list = []

        x = 0
        for k in I:
            y = 0
            for l in I:
                if I[x, y] > self.i_threshold:
                    x_start = int(x - (self.a_roi - 1)/2)
                    y_start = int(y - (self.a_roi - 1)/2)
                    x_end = int(x_start + self.a_roi)
                    y_end = int(y_start + self.a_roi)
            # 2.
                    if (x_start < 0 or y_start < 0 or
                            x_end > len(I)-1 or y_end > len(I.T)-1):
                        y += 1
                        continue
            # 3.
                    i_bottom = self.sum_border(x_start, x_end-1,
                                               'row', y_start, I)
                    i_top = self.sum_border(x_start+1, x_end,
                                            'row', y_end, I)
                    i_left = self.sum_border(y_start, y_end-1,
                                             'column', x_start, I)
                    i_right = self.sum_border(y_start+1, y_end,
                                              'column', x_end, I)
                    i_border = ((i_top + i_bottom + i_left + i_right) /
                                4 * (self.a_roi - 1))

            # 4. - normalization
                    I_norm_matrix = I - i_border

            # 5.
                    b = self.total_mass(x_start+1, x_end-1,
                                        y_start+1, y_end-1, I_norm_matrix)

                    x_cm, y_cm = self.array_point_mass(x_start+1, x_end-1,
                                                       y_start+1, y_end-1,
                                                       I_norm_matrix, b)

                    star_list.append([x_cm, y_cm])
                y += 1
            x += 1

        clustering = True
        if clustering:
            # 6. Clustering
            control = 1
            pixel_diff = 15
            while control > 0:
                star_list2 = star_list
                control = 0
                for star1, star2 in itertools.combinations(star_list2, 2):
                    if (star2[0] + pixel_diff >= star1[0] >=
                            star2[0] - pixel_diff and
                            star2[1] + pixel_diff >= star1[1] >=
                            star2[1] - pixel_diff):
                        control += 1

                        star_list.remove(star1)
                        star_list.remove(star2)
                        star_list.append([(star1[0] + star2[0]) / 2,
                                          (star1[1] + star2[1]) / 2])
                        break

        vectors_list = []

        # 7. unit vector u
        for star in star_list:
            vector = np.array([self.pixel_size * star[0],
                               self.pixel_size * star[1],
                               self.focal_length])
            u = vector.T / np.linalg.norm(vector)
            vectors_list.append(u)

        img = np.zeros((len(I), len(I.T)), dtype='uint8')
        for star in star_list:
            img[int(star[0]), int(star[1])] = 255
        Image.fromarray(img, mode='L').convert('1').save('test.png')

        return star_list
