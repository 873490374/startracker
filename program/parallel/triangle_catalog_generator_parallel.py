import csv
import operator
import numpy as np

from numba import autojit
from progress.bar import Bar
from timeit import default_timer as timer

from program.planar_triangle import ImagePlanarTriangle
from program.star import StarPosition
from program.tracker.kvector_calculator import KVectorCalculator
from program.parallel.planar_triangle_calculator_np import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
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

"""


class TriangleCatalogGeneratorParallel:
    def __init__(
            self, max_magnitude: int, sensor_variance: float, camera_fov: int):
        self.star_identifier = StarIdentifier(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=sensor_variance
            ),
            kvector_calculator=KVectorCalculator(),
            max_magnitude=max_magnitude,
            camera_fov=camera_fov,
            catalog=None)
        self.kvector_calc = KVectorCalculator()
        self.triangle_calc = PlanarTriangleCalculator(
            sensor_variance=sensor_variance)
        self.max_magnitude = max_magnitude
        self.sensor_variance = sensor_variance
        self.camera_fov = camera_fov

    def generate_triangles(
            self, star_catalog_path: str) -> [ImagePlanarTriangle]:
        converted_stars = np.array([], dtype=np.float64)
        triangle_catalogue = np.array([], dtype=np.float64)

        # converted_stars = []
        # triangle_catalogue = []

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s.magnitude <= self.max_magnitude:
                star = convert_star_to_uv(s)
                star = self.convert_star_to_np(star)
                # converted_stars.append(star)
                if len(converted_stars) == 0:
                    converted_stars = np.hstack((converted_stars, star))
                    continue
                converted_stars = np.vstack((converted_stars, star))
        print('Building planar triangle catalogue')
        i = 0
        bar1 = Bar('s1', max=len(converted_stars))
        start = timer()
        triangle_catalogue = self.method_name(bar1, converted_stars, i,
                                              triangle_catalogue)
        dt = timer() - start
        bar1.finish()
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalogue)))
        print("time: {}".format(dt))
        triangle_catalogue = self.sort_catalog(triangle_catalogue)
        triangle_catalogue = self.add_k_vector(triangle_catalogue)
        return triangle_catalogue

    # @autojit
    def method_name(self, bar1, converted_stars, i, triangle_catalogue):
        for s1 in converted_stars:
            bar1.next()
            i += 1
            j = i
            triangle_catalogue = self.s2_s3_triangles(
                converted_stars, i, j, s1, triangle_catalogue)
        return triangle_catalogue

    # @autojit
    def s2_s3_triangles(self, converted_stars: np.ndarray, i: int, j: int, s1: np.ndarray, triangle_catalogue: np.ndarray):
        bar2 = Bar('s2', max=len(converted_stars[i:]))
        for s2 in converted_stars[i:]:
            bar2.next()
            j += 1
            # bar3 = Bar('s3', max=len(converted_stars[j:]))
            for s3 in converted_stars[j:]:
                # bar3.next()
                triangle_catalogue = self.create_triangles(
                    s1, s2, s3, triangle_catalogue)
            # bar3.finish()
        bar2.finish()
        return triangle_catalogue

    # @autojit
    def create_triangles(self, s1, s2, s3, triangle_catalogue):
        if are_stars_valid(s1, s2, s3, self.max_magnitude, self.camera_fov):
            triangle = self.triangle_calc.calculate_triangle(
                s1, s2, s3)
            if len(triangle_catalogue) == 0:
                triangle_catalogue = np.hstack((triangle_catalogue, triangle))
                return triangle_catalogue
            triangle_catalogue = np.vstack((triangle_catalogue, triangle))
        return triangle_catalogue

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
        return sorted(catalog, key=operator.attrgetter('moment'))

    def add_k_vector(self, catalog):
        k_vector, _, _ = self.kvector_calc.make_kvector(catalog)
        return k_vector

    def save_to_file(
            self, catalog: [ImagePlanarTriangle], output_file_path: str):

        with open(output_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=[
                'star1_id', 'star2_id', 'star3_id', 'area', 'moment', 'k'])
            csvwriter.writeheader()
            for t in catalog:
                csvwriter.writerow(dict(t))

    def convert_star_to_np(self, star):
        s = np.array([star.id, star.magnitude, star.unit_vector[0],
                  star.unit_vector[1], star.unit_vector[2]])
        return s


# @autojit
def are_stars_valid(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray,
                    max_magnitude: float, camera_fov: int) -> bool:
    if any([
        s1[2] == s2[2] and s1[3] == s2[3] and s1[4] == s2[4],
        s1[2] == s3[2] and s1[3] == s3[3] and s1[4] == s3[4],
        s3[2] == s2[2] and s3[3] == s2[3] and s3[4] == s2[4],
    ]):
        return False
    if not all([
        s1[1] <= max_magnitude,
        s2[1] <= max_magnitude,
        s3[1] <= max_magnitude,
    ]):
        return False
    uv1 = np.array([s1[2], s1[3], s1[4]])
    uv2 = np.array([s2[2], s2[3], s2[4]])
    uv3 = np.array([s3[2], s3[3], s3[4]])
    return all([
        np.inner(uv1.T, uv2) >= np.cos(np.deg2rad(camera_fov)),
        np.inner(uv2.T, uv3) >= np.cos(np.deg2rad(camera_fov)),
        np.inner(uv1.T, uv3) >= np.cos(np.deg2rad(camera_fov)),
    ])
