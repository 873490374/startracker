import csv
import datetime

import numpy as np
from progress.bar import Bar

from program.planar_triangle import CatalogPlanarTriangle
from program.star import StarPosition, StarUV
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.validation.scripts.simulator import StarCatalog


class CatalogGenerator:
    def __init__(self):
        self.star_identifier = StarIdentifier(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=SENSOR_VARIANCE
            ),
            sensor_variance=SENSOR_VARIANCE,
            max_magnitude=MAX_MAGNITUDE,
            camera_fov=CAMERA_FOV,
            catalog=None)
        self.triangle_calc = PlanarTriangleCalculator(
            sensor_variance=SENSOR_VARIANCE)

    def generate_triangles(self) -> [CatalogPlanarTriangle]:
        converted_start = []
        triangle_catalogue = []

        stars = self.read_catalogue_stars()
        for s in stars:
            if s.magnitude <= MAX_MAGNITUDE:
                star = self.convert(s)
                converted_start.append(star)
        print(len(converted_start))
        classify_bar = Bar(
            'Building planar triangle catalogue', max=len(converted_start))
        i = 0
        for s1 in converted_start:
            i += 1
            j = 0
            classify_bar.next()
            for s2 in converted_start[i:]:
                j += 1
                for s3 in converted_start[i+j:]:
                    if self.star_identifier.are_stars_valid(
                            s1, s2, s3, MAX_MAGNITUDE, CAMERA_FOV):
                        triangle = self.triangle_calc.calculate_triangle(
                            s1.unit_vector,
                            s2.unit_vector,
                            s3.unit_vector,
                        )
                        triangle_catalogue.append(triangle)
        classify_bar.finish()
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalogue)))
        # kvector
        self.save_to_file(triangle_catalogue)
        return triangle_catalogue

    def read_catalogue_stars(self) -> [StarPosition]:
        stars = []
        import os

        catalog = StarCatalog(
            '{}/program/validation/data/hip_main.dat'.format(os.getcwd())).catalog
        for row in catalog.itertuples():
            s = StarPosition(
                row[2],
                row[6],
                row[9],
                row[10],
            )
            stars.append(s)
        return stars

    def convert(self, star: StarPosition) -> StarUV:
        """ Convert star positions to unit vector."""
        alpha = star.right_ascension
        delta = star.declination
        return StarUV(
            star_id=star.id,
            magnitude=star.magnitude,
            unit_vector=np.array([
                np.cos(alpha) * np.cos(delta),
                np.sin(alpha) * np.cos(delta),
                np.sin(delta)
            ], dtype='float64').T
        )

    def save_to_file(self, catalog: [CatalogPlanarTriangle]):
        now = datetime.datetime.now()
        with open(
                './program/catalog/generated/triangle_catalog_'
                '{}_{}_{}_{}_{}.csv'.format(
                    now.year, now.month, now.day, now.hour, now.minute),
                'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=[
                'star1_id', 'star2_id', 'star3_id', 'area', 'polar_moment'])
            for t in catalog:
                # print(t)
                # print(dict(t))
                csvwriter.writerow(dict(t))


generator = CatalogGenerator()
generator.generate_triangles()
