import csv
import datetime
import operator
import os

import numpy as np
from numba import *
from numba import cuda
from progress.bar import Bar
from timeit import default_timer as timer

from program.const import MAIN_PATH
from program.planar_triangle import ImagePlanarTriangle
from program.star import StarPosition
from program.tracker.kvector_calculator import KVectorCalculator
from program.parallel.planar_triangle_calculator_np import \
    calculate_triangle
from program.utils import convert_star_to_uv
from program.validation.scripts.simulator import StarCatalog


"""
magnitude: 3
first jit
time: 323.67861194199986

second jit
time: 48.604901504000736

second no jit
time: 17.909163428000284

second no jit
time: 6.240033824999955

"""


class TriangleCatalogGeneratorParallel:
    def __init__(
            self, max_magnitude: int, sensor_variance: float, camera_fov: int):
        self.kvector_calc = KVectorCalculator()
        self.max_magnitude = max_magnitude
        self.sensor_variance = sensor_variance
        self.camera_fov = camera_fov

    def generate_triangles(
            self, star_catalog_path: str) -> [ImagePlanarTriangle]:
        converted_stars = np.array([], dtype=np.float64)

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s.magnitude <= self.max_magnitude:
                star = convert_star_to_uv(s)
                # TODO
                star = self.convert_star_to_np(star)
                # converted_stars.append(star)
                if len(converted_stars) == 0:
                    converted_stars = np.hstack((converted_stars, star))
                    continue
                converted_stars = np.vstack((converted_stars, star))
        print('Building planar triangle catalogue')
        i = 0

        timestamp = datetime.datetime.now()
        start = timer()
        self.calculate_triangles(timestamp, converted_stars, i)
        dt = timer() - start
        print("time: {}".format(dt))
        triangle_catalog = self.put_triangles_parts_together(
            timestamp, len(converted_stars))
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalog)))

        triangle_catalog = self.sort_catalog(triangle_catalog)
        # TODO kvector to numpy
        triangle_catalog = self.add_k_vector(triangle_catalog)
        return triangle_catalog

    def calculate_triangles(self, timestamp, converted_stars, i):
        bar1 = Bar('s1', max=len(converted_stars))
        blockdim = (32, 8)
        griddim = (32, 16)
        i = 0

        d_stars = cuda.to_device(converted_stars)
        for s1 in converted_stars:
            bar1.next()
            max_one_time_triangles = 500
            triangles = np.zeros((max_one_time_triangles, 5), dtype=np.float64)
            i += 1
            d_s1 = cuda.to_device(s1)
            d_catalog = cuda.to_device(triangles)
            triangle_kernel[griddim, blockdim](d_s1, i, d_stars, d_catalog)
            d_catalog.to_host()

            self.get_save_triangles_part(
                timestamp, i, max_one_time_triangles, triangles)
        bar1.finish()
        return

    def get_save_triangles_part(
            self, dtime, part_nr, max_one_time_triangles, triangles):
        catalog = np.array([], dtype=np.float64)
        for j in range(max_one_time_triangles):
            t = triangles[j]
            if t[0] == 0:
                continue
            if len(catalog) == 0:
                catalog = np.hstack((catalog, t))
                continue
            catalog = np.vstack((catalog, t))
        self.save_partially_to_file(catalog, part_nr, dtime)

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

    def sort_catalog(self, catalog):
        return catalog[catalog[:, 4].argsort()]

    def add_k_vector(self, catalog):
        k_vector, _, _ = self.kvector_calc.make_kvector(catalog)
        return k_vector

    def save_to_file(
            self, catalog: np.ndarray, output_file_path: str):
        np.savetxt(output_file_path, catalog, delimiter=',')

    def convert_star_to_np(self, star):
        s = np.array([star.id, star.magnitude, star.unit_vector[0],
                  star.unit_vector[1], star.unit_vector[2]])
        return s

    def save_partially_to_file(
            self, catalog: np.ndarray, part_nr: int, dtime: datetime.datetime):
        output_file_path = os.path.join(
            MAIN_PATH, './program/catalog/generated/triangle_catalog_partial_'
                       '{}_{}_{}.{}'.format(
                dtime.year, dtime.month, dtime.day, part_nr))

        np.savetxt(output_file_path, catalog, delimiter=',')

    def put_triangles_parts_together(self, timestamp, parts_amount):
        # parts_amount = 8
        alist = []

        output_file_path = os.path.join(
            MAIN_PATH, './program/catalog/generated/'
                       'triangle_catalog_full_{}_{}_{}.csv'.format(
                timestamp.year, timestamp.month, timestamp.day))

        for i in range(1, parts_amount):

            input_file_path = os.path.join(
                MAIN_PATH, './program/catalog/generated/'
                           'triangle_catalog_partial_{}_{}_{}.{}'.format(
                    timestamp.year, timestamp.month, timestamp.day, i))

            with open(input_file_path, 'rb') as f:
                triangles = np.genfromtxt(f, dtype=np.float64, delimiter=',')
                alist.append(triangles)
        triangle_list = np.concatenate(alist)
        return triangle_list


triangle_gpu = cuda.jit(
    restype=(float64, float64),
    argtypes=[float64[:], float64[:], float64[:]],
    device=True)(calculate_triangle)


@cuda.jit(argtypes=[float64[:], int8, float64[:, :], float64[:, :]])
def triangle_kernel(s1, i, stars, catalog):
    n = len(stars)
    k = 0
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    j = i
    for x in range(i, n, gridX):
        s2 = stars[x]
        j += 1
        for y in range(j, n, gridY):
            s3 = stars[y]
            # print('hello')
            area, moment = triangle_gpu(s1, s2, s3)
            if area is None:
                continue
            catalog[k][0] = s1[0]
            catalog[k][1] = s2[0]
            catalog[k][2] = s3[0]

            catalog[k][3] = area
            catalog[k][4] = moment
            k += 1
            if k > len(catalog)-1:
                print('out of memory')
