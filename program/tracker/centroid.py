import itertools
import numpy as np
from PIL import Image
from numba import cuda, float64, int32

from program.star import StarUV


A_ROI = 5
I_THRESHOLD = 250


class CentroidCalculator:
    def __init__(self, pixel_size: int, focal_length: int,
                 a_roi: int, i_threshold: int):
        self.pixel_size = pixel_size
        self.focal_length = focal_length
        self.a_roi = a_roi
        self.i_threshold = i_threshold

    def calculate_centroids(self, I: np.ndarray) -> [StarUV]:

        star_list = self.preprocess_image_matrix(I)

        star_list = self.clustering(star_list)

        star_vectors = self.convert_to_vectors(star_list)

        self.create_image(I, star_list)

        return star_vectors

    def create_image(self, I, star_list):
        img = np.zeros((len(I), len(I.T)), dtype='uint8')
        for star in star_list:
            if star[0] > len(I) or star[1] > len(I):
                print('why')
            img[int(star[0]), int(star[1])] = 255
        Image.fromarray(img, mode='L').convert('1').save('test.png')

    def convert_to_vectors(self, star_list):
        star_vectors = []
        # 7. unit vector u
        for star in star_list:
            # TODO Does it work correctly? What are focal_length & pixel size?
            vector = np.array([self.pixel_size * star[0],
                               self.pixel_size * star[1],
                               self.focal_length])
            u = vector.T / np.linalg.norm(vector)
            star_vectors.append(
                StarUV(star_id=-1, magnitude=-1, unit_vector=u))
        return star_vectors

    def preprocess_image_matrix(self, I):

        x_size, y_size, = I.shape

        calc_img = np.zeros((x_size, y_size, 2), dtype=np.float64)

        blockdim = (32, 8)
        griddim = (32, 16)
        d_img = cuda.to_device(I.astype(np.float64))
        d_calc = cuda.to_device(calc_img)
        preprocess_kernel[griddim, blockdim](d_img, d_calc, x_size, y_size)
        calc_img = d_calc.copy_to_host()

        # calc_img = preprocess_no_gpu(I, calc_img, x_size, y_size)

        star_list = self.calc_img_to_star_list(calc_img)

        return star_list

    def calc_img_to_star_list(self, calc_img):
        star_list = calc_img[~(calc_img == 0).all(2)]
        return star_list.tolist()

    def clustering(self, star_list):
        # 6. Clustering
        control = 1
        pixel_diff = 5
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
        I: np.ndarray, x: int, y: int, x_size, y_size):
    if I[x, y] > I_THRESHOLD:
        x_start = int(x - (A_ROI - 1) / 2)
        y_start = int(y - (A_ROI - 1) / 2)
        x_end = int(x_start + A_ROI)
        y_end = int(y_start + A_ROI)
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
                    4 * (A_ROI - 1))
        cuda.syncthreads()
        # 4. - normalization
        # TODO make normalization matrix not randomizing results
        I_norm_matrix = I
        for x_norm in range(x_start + 1, x_end):
            for y_norm in range(y_start + 1, y_end):
                I_norm_matrix[x_norm, y_norm] = I[x_norm, y_norm] - i_border
        cuda.syncthreads()
        # 5.
        # total mass
        b = 0
        for xb in range(x_start + 1, x_end - 1):
            for yb in range(y_start + 1, y_end - 1):
                b += I_norm_matrix[xb, yb]
        cuda.syncthreads()
        # array point mass
        x_cm = 0
        y_cm = 0
        for xm in range(x_start + 1, x_end - 1):
            for ym in range(y_start + 1, y_end - 1):
                x_cm += I_norm_matrix[xm, ym] * xm / b
                y_cm += I_norm_matrix[xm, ym] * ym / b
        cuda.syncthreads()
        return x_cm, y_cm
    return 0, 0


pixel_preprocess_gpu = cuda.jit(
    restype=(float64, float64),
    argtypes=[float64[:, :], int32, int32, int32, int32],
    device=True)(pixel_preprocess)


@cuda.jit(argtypes=[float64[:, :], float64[:, :, :], int32, int32])
def preprocess_kernel(image, calc_img, x_size, y_size):

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, x_size, gridX):
        for y in range(startY, y_size, gridY):
            x_calc, y_calc = pixel_preprocess_gpu(image, x, y, x_size, y_size)

            calc_img[x][y][0] = x_calc
            calc_img[x][y][1] = y_calc

#
# def preprocess_no_gpu(image, calc_img, x_size, y_size):
#     for x in range(0, x_size):
#         for y in range(0, y_size):
#             x_calc, y_calc = pixel_preprocess(image, x, y, x_size, y_size)
#
#             calc_img[x][y][0] = x_calc
#             calc_img[x][y][1] = y_calc
#     return calc_img
