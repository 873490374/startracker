import datetime
import os
import warnings

import numpy as np

from numba import cuda, float64, int8
from progress.bar import Bar
from timeit import default_timer as timer

from program.const import MAIN_PATH
from program.star import StarPosition
from program.tracker.kvector_calculator import KVectorCalculator
from program.catalog.planar_triangle_calculator_np import (
    calculate_catalog_triangle,
)
from program.validation.scripts.simulator import StarCatalog


class TriangleCatalogGeneratorParallel:
    def __init__(
            self, max_magnitude: int):
        self.kvector_calc = KVectorCalculator()
        self.max_magnitude = max_magnitude

    def generate_triangles(
            self, star_catalog_path: str) -> (np.ndarray, float, float):
        converted_stars = np.array([], dtype=np.float64)

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s.magnitude <= self.max_magnitude:
                star = self.convert_star_to_np(s)
                if len(converted_stars) == 0:
                    converted_stars = np.hstack((converted_stars, star))
                    continue
                converted_stars = np.vstack((converted_stars, star))
        print('Building planar triangle catalogue')

        timestamp = datetime.datetime.now()
        start = timer()
        self.calculate_triangles(timestamp, converted_stars)
        dt = timer() - start
        print("time: {}".format(dt))
        triangle_catalog = self.put_triangles_parts_together(
            timestamp, len(converted_stars))
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalog)))

        triangle_catalog = self.sort_catalog(triangle_catalog)
        triangle_catalog, m, q = self.add_k_vector(triangle_catalog)
        return triangle_catalog, m, q

    def calculate_triangles(self, timestamp, converted_stars):
        bar1 = Bar('s1', max=len(converted_stars))
        blockdim = (32, 8)
        griddim = (32, 16)
        i = 0

        d_stars = cuda.to_device(converted_stars)
        for s1 in converted_stars:
            bar1.next()

            triangles = np.zeros(
                (len(converted_stars), len(converted_stars), 4),
                dtype=np.float64)
            i += 1

            d_s1 = cuda.to_device(s1)
            d_catalog = cuda.to_device(triangles)
            triangle_kernel[griddim, blockdim](d_s1, i, d_stars, d_catalog)
            triangles = d_catalog.copy_to_host()
            self.save_triangles_part(s1[0], timestamp, i, triangles)
        bar1.finish()
        return

    def save_triangles_part(
            self, s1_id, dtime, part_nr, triangles):
        # Reshape from rectangular 3D matrix to 2D matrix (array) of triangles
        tr = triangles.reshape(-1, 4)

        # Remove empty triangles
        tr = tr[tr[:, 3] > 0.]
        # tr = tr[tr[:, 0] != tr[:, 1]]
        # tr = tr[tr[:, 0] != s1_id]
        # tr = tr[tr[:, 1] != s1_id]

        # Add star1 id in the new first column
        t = np.zeros((len(tr), 5))
        t[:, 1:] = tr
        t[:, 0] = s1_id
        t = self.remove_duplicates(t)
        self.save_partially_to_file(t, part_nr, dtime)

    def read_catalogue_stars(self, star_catalog_path: str) -> [StarPosition]:
        stars = []

        catalog = StarCatalog(self.max_magnitude, star_catalog_path).catalog
        for row in catalog.itertuples():
            s = StarPosition(
                row[2],
                row[6],
                row[9],
                row[10],
            )
            stars.append(s)
        return stars

    @staticmethod
    def sort_catalog(catalog):
        return catalog[catalog[:, 4].argsort()]

    def add_k_vector(self, catalog):
        k_vector, m, q = self.kvector_calc.make_kvector(catalog)
        return k_vector, m, q

    @staticmethod
    def save_to_file(
            catalog: np.array, output_file_path: str):
        np.savetxt(output_file_path, catalog, delimiter=',')

    @staticmethod
    def save_m_q_to_file(m, q, output_file_path):
        with open(output_file_path, 'w') as f:
            f.writelines([str(m)+'\n', str(q)])

    @staticmethod
    def convert_star_to_np(star: StarPosition) -> np.ndarray:

        alpha = np.deg2rad(star.right_ascension)
        delta = np.deg2rad(star.declination)

        s = np.array([
            star.id, star.magnitude,
            np.array([np.cos(alpha) * np.cos(delta)]),
            np.array([np.sin(alpha) * np.cos(delta)]),
            np.array([np.sin(delta)]),
        ])
        return s

    @staticmethod
    def save_partially_to_file(
            catalog: np.array, part_nr: int, dtime: datetime.datetime):
        output_file_path = os.path.join(
            MAIN_PATH,
            './program/catalog/generated/triangle_catalog_partial_'
            '{}_{}_{}.{}'.format(
                dtime.year, dtime.month, dtime.day, part_nr))

        np.savetxt(output_file_path, catalog, delimiter=',')

    def put_triangles_parts_together(self, timestamp, parts_amount):
        catalog = np.empty([1, 5], dtype=np.float64)
        c = []

        bar = Bar('Reading triangles parts from files', max=parts_amount)
        for i in range(1, parts_amount + 1):
            bar.next()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_file_path = os.path.join(
                    MAIN_PATH,
                    './program/catalog/generated/'
                    'triangle_catalog_partial_{}_{}_{}.{}'.format(
                        timestamp.year, timestamp.month, timestamp.day, i))

                with open(input_file_path, 'rb') as f:
                    triangles = np.genfromtxt(
                        f, dtype=np.float64, delimiter=',')
                    c.append(triangles)
        bar.finish()

        bar2 = Bar('Putting triangles together', max=parts_amount)
        for triangles in c:
            bar2.next()
            catalog = self.append_to_table(catalog, triangles)
        catalog = np.delete(catalog, 0, axis=0)
        bar2.finish()

        catalog = self.remove_duplicates(catalog)
        return catalog

    @staticmethod
    def remove_duplicates(tr):
        trc1 = np.copy(tr[:, 0:3])
        trc1 = np.sort(trc1)
        x = np.random.rand(trc1.shape[1])
        y = trc1.dot(x)
        _, index = np.unique(y, return_index=True)

        tr = tr[index]
        trc1 = np.copy(tr[:, 0:3])
        trc1 = np.sort(trc1)
        tr[:, 0:3] = trc1[:, 0:3]

        return tr

    def append_to_table(self, table, rows):
        if rows.size > 0 and rows.ndim > 1:
            return np.append(table, rows, axis=0)
        elif rows.size > 0 and rows.ndim == 1:
            return self.add_to_table(table, rows)
        return table

    @staticmethod
    def add_to_table(table, row):
        if len(table) == 0:
            return np.hstack((table, row))
        return np.vstack((table, row))


triangle_gpu = cuda.jit(
    restype=(float64, float64),
    argtypes=[float64[:], float64[:], float64[:]],
    device=True)(calculate_catalog_triangle)


@cuda.jit(argtypes=[float64[:], int8, float64[:, :], float64[:, :, :]])
def triangle_kernel(s1, i, stars, catalog):
    n = len(stars)
    j = i
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    start_x = max(startX, i)
    for x in range(start_x, n, gridX):
        s2 = stars[x]
        j += 1
        start_y = max(startY, j)
        for y in range(start_y, n, gridY):
            s3 = stars[y]
            area, moment = triangle_gpu(s1, s2, s3)

            catalog[x][y][0] = s2[0]
            catalog[x][y][1] = s3[0]
            catalog[x][y][2] = area
            catalog[x][y][3] = moment
