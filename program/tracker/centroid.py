import itertools

import numpy as np
from PIL import Image
from numba import cuda, float64, int32

from program.utils import convert_to_vector


class CentroidCalculator:
    def __init__(self, pixel_size: int, focal_length: int,
                 a_roi: int, clustering_roi: int, i_threshold: int,
                 mag_threshold: int, star_mag_pix: int,
                 principal_point: (int, int)):
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.a_roi = a_roi
        self.c_roi = clustering_roi
        self.i_threshold = i_threshold
        self.mag_threshold = mag_threshold
        self.star_mag_pix = star_mag_pix
        self.principal_point = principal_point

    def calculate_centroids(self, I: np.ndarray) -> [np.ndarray]:

        star_list = self.preprocess_image_matrix(I)
        star_list = self.clustering(star_list)
        star_vectors = self.convert_to_vectors(star_list)

        # self.create_image(I, star_list)

        return star_vectors

    def preprocess_image_matrix(self, I):

        x_size, y_size, = I.shape

        calc_img = np.zeros((x_size, y_size, 2), dtype=np.float64)

        I_norm = np.zeros(
            (x_size, y_size, self.a_roi-2, self.a_roi-2), dtype=np.float64)

        blockdim = (16, 8)
        griddim = (32, 16)
        d_img = cuda.to_device(I.astype(np.float64))
        d_calc = cuda.to_device(calc_img)
        d_norm = cuda.to_device(I_norm)
        preprocess_kernel[griddim, blockdim](
            d_img, d_calc, d_norm, x_size, y_size,
            self.i_threshold, self.mag_threshold,
            self.star_mag_pix, self.a_roi)
        calc_img = d_calc.copy_to_host()

        # calc_img = preprocess_no_gpu(
        #     I, calc_img, I_norm, x_size, y_size,
        #     self.i_threshold, self.mag_threshold,
        #     self.star_mag_pix, self.a_roi)

        star_list = self.calc_img_to_star_list(calc_img)

        return star_list

    def convert_to_vectors(self, star_list):
        star_vectors = []
        # 7. unit vector u
        i = 0
        for star in star_list:
            u = convert_to_vector(
                star[0], star[1], self.pixel_size,
                self.focal_length, self.principal_point)
            # FIXME check if all x-y coordinates all correct till this point
            star_vectors.append(
                np.array([i, u[0], u[1], u[2], star[1], star[0]]))
            i += 1
        return star_vectors

    @staticmethod
    def create_image(I, star_list):
        img = np.zeros((len(I), len(I.T)), dtype='uint8')
        for star in star_list:
            x = int(star[0])
            y = int(star[1])
            img[x, y] = 255
            img[x-1, y] = 255
            img[x+1, y] = 255
            img[x, y-1] = 255
            img[x, y+1] = 255
        Image.fromarray(img, mode='L').convert('1').save('test.png')

    @staticmethod
    def calc_img_to_star_list(calc_img):
        star_list = calc_img[~(calc_img == 0).all(2)]
        return star_list.tolist()

    def clustering(self, star_list):
        # 6. Clustering
        control = 1
        pixel_diff = self.c_roi
        while control > 0:
            star_list2 = star_list
            control = 0
            for star1, star2 in itertools.combinations(star_list2, 2):
                if (
                        star2[0] + pixel_diff >= star1[0] >=
                        star2[0] - pixel_diff and
                        star2[1] + pixel_diff >= star1[1] >=
                        star2[1] - pixel_diff
                ):
                    control += 1

                    star_list.remove(star1)
                    star_list.remove(star2)
                    star_list.append((
                        (star1[0] + star2[0]) / 2,
                        (star1[1] + star2[1]) / 2))
                    break
        return star_list


def pixel_preprocess(
        I: np.ndarray, I_norm_matrix: np.ndarray,
        x: int, y: int, x_size, y_size, i_threshold,
        mag_threshold, star_mag_pix, a_roi):
    if I[x, y] > i_threshold:
        half = int(star_mag_pix/2)
        if (x < half or x > x_size - half or
                y < half or y > y_size - half):
            return 0, 0
        sum_ = 0
        sum_ += I[x, y]
        pixels = 1
        for i in range(half):
            sum_ += I[x + i, y + i]
            sum_ += I[x - i, y - i]
            sum_ += I[x + i, y]
            sum_ += I[x - i, y]
            sum_ += I[x, y + i]
            sum_ += I[x, y - i]
            pixels += 6
        if sum_/pixels < mag_threshold:
            return 0, 0
        x_start = int(x - (a_roi - 1) / 2)
        y_start = int(y - (a_roi - 1) / 2)
        x_end = int(x_start + a_roi)
        y_end = int(y_start + a_roi)
        # 2.
        if (x_start < 0 or y_start < 0 or
                x_end > x_size - 1 or y_end > y_size - 1):
            return 0, 0

        # 3.
        i_bottom = 0
        for i in range(x_start, x_end - 1):
            i_bottom += I[i, y_end]

        i_top = 0
        for i in range(x_start + 1, x_end):
            i_top += I[i, y_start]

        i_left = 0
        for i in range(y_start, y_end - 1):
            i_left += I[x_start, i]

        i_right = 0
        for i in range(y_start + 1, y_end):
            i_right += I[x_end, i]

        i_border = ((i_top + i_bottom + i_left + i_right) /
                    4 * (a_roi - 1))
        # 4. - normalization
        x_nn = 0
        for x_norm in range(x_start+1, x_end-1):
            y_nn = 0
            for y_norm in range(y_start+1, y_end-1):
                I_norm_matrix[x, y, x_nn, y_nn] = I[x_norm, y_norm] - i_border
                y_nn += 1
            x_nn += 1
        # 5.
        # total mass
        b = 0
        x_nn = 0
        for xb in range(x_start + 1, x_end - 1):
            y_nn = 0
            for yb in range(y_start + 1, y_end - 1):
                b += I_norm_matrix[x, y, x_nn, y_nn]
                y_nn += 1
            x_nn += 1
        # array point mass
        x_cm = 0
        y_cm = 0
        x_nn = 0
        for xm in range(x_start + 1, x_end - 1):
            y_nn = 0
            for ym in range(y_start + 1, y_end - 1):
                x_cm += I_norm_matrix[x, y, x_nn, y_nn] * xm / b
                y_cm += I_norm_matrix[x, y, x_nn, y_nn] * ym / b
                y_nn += 1
            x_nn += 1
        return x_cm, y_cm
    return 0, 0


pixel_preprocess_gpu = cuda.jit(
    restype=(float64, float64),
    argtypes=[
        float64[:, :], float64[:, :, :, :],
        int32, int32, int32, int32, int32, int32, int32, int32],
    device=True)(pixel_preprocess)


@cuda.jit(argtypes=[
    float64[:, :], float64[:, :, :], float64[:, :, :, :],
    int32, int32, int32, int32, int32, int32])
def preprocess_kernel(
        image, calc_img, img_norm, x_size, y_size,
        i_threshold, mag_threshold, star_mag_pix, a_roi):

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, x_size, grid_x):
        for y in range(start_y, y_size, grid_y):
            x_calc, y_calc = pixel_preprocess_gpu(
                image, img_norm, x, y, x_size, y_size,
                i_threshold, mag_threshold, star_mag_pix, a_roi)

            calc_img[x][y][0] = x_calc
            calc_img[x][y][1] = y_calc


def preprocess_no_gpu(
        image, calc_img, img_norm, x_size, y_size,
        i_threshold, mag_threshold, star_mag_pix, a_roi):
    for x in range(0, x_size):
        for y in range(0, y_size):
            x_calc, y_calc = pixel_preprocess(
                image, img_norm, x, y, x_size, y_size,
                i_threshold, mag_threshold, star_mag_pix, a_roi)

            calc_img[x][y][0] = x_calc
            calc_img[x][y][1] = y_calc
    return calc_img
