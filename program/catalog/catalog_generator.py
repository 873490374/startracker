import csv
import operator

import numpy as np
from progress.bar import Bar
from timeit import default_timer as timer

from program.const import COS_CAMERA_FOV
from program.planar_triangle import ImagePlanarTriangle
from program.star import StarPosition, StarUV
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.utils import convert_star_to_uv
from program.validation.scripts.simulator import StarCatalog


"""
magnitude: 3
time: 234.87970195999992
"""


class CatalogGenerator:
    def __init__(
            self, max_magnitude: int, sensor_variance: float, camera_fov: int):
        self.kvector_calc = KVectorCalculator()
        self.triangle_calc = PlanarTriangleCalculator(
            sensor_variance=sensor_variance)
        self.max_magnitude = max_magnitude
        self.sensor_variance = sensor_variance
        self.camera_fov = camera_fov

    def generate_triangles(
            self, star_catalog_path: str) -> [ImagePlanarTriangle]:
        converted_stars = []
        triangle_catalogue = []

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s.magnitude <= self.max_magnitude:
                star = convert_star_to_uv(s)
                converted_stars.append(star)
        print('Building planar triangle catalogue')
        i = 0
        bar1 = Bar('s1', max=len(converted_stars))
        start = timer()
        for s1 in converted_stars:
            bar1.next()
            i += 1
            j = i
            # bar2 = Bar('s2', max=len(converted_stars[i:]))
            for s2 in converted_stars[i:]:
                # bar2.next()
                j += 1
                # bar3 = Bar('s3', max=len(converted_stars[j:]))
                for s3 in converted_stars[j:]:
                    # bar3.next()
                    if self.are_stars_valid(
                            s1, s2, s3):
                        triangle = self.triangle_calc.calculate_triangle(
                            s1, s2, s3)
                        triangle_catalogue.append(triangle)
                # bar3.finish()
            # dt = timer() - start
            # print("time: {}".format(dt))
            # bar2.finish()
        dt = timer() - start
        bar1.finish()
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalogue)))
        print("time: {}".format(dt))
        triangle_catalogue = self.sort_catalog(triangle_catalogue)
        triangle_catalogue = self.add_k_vector(triangle_catalogue)
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

    def are_stars_valid(self, s1: StarUV, s2: StarUV, s3: StarUV) -> bool:
        l1 = s1.unit_vector[0] * s2.unit_vector[0] + \
             s1.unit_vector[1] * s2.unit_vector[1] + \
             s1.unit_vector[2] * s2.unit_vector[2]
        l2 = s2.unit_vector[0] * s3.unit_vector[0] + \
             s2.unit_vector[1] * s3.unit_vector[1] + \
             s2.unit_vector[2] * s3.unit_vector[2]
        l3 = s1.unit_vector[0] * s3.unit_vector[0] + \
             s1.unit_vector[1] * s3.unit_vector[1] + \
             s1.unit_vector[2] * s3.unit_vector[2]
        if (
                l1 >= COS_CAMERA_FOV and
                l2 >= COS_CAMERA_FOV and
                l3 >= COS_CAMERA_FOV
        ):
            return True
        return False
