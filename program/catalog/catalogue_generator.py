import csv
import datetime

import numpy as np
from progress.bar import Bar

from program.classes import CatalogueTriangle, StarPosition, StarUV
from program.const import SENSOR_VARIANCE, MAX_MAGNITUDE, CAMERA_FOV
from program.tracker.planar_triangle import PlanarTriangle
from program.tracker.star_identifier import is_valid
from program.validation.scripts.simulator import StarCatalog


class CatalogueGenerator:

    def generate_triangles(self) -> [CatalogueTriangle]:
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
                    if is_valid(s1, s2, s3, MAX_MAGNITUDE, CAMERA_FOV):
                        triangle = PlanarTriangle()
                        triangle.calculate_triangle(
                            s1.unit_vector,
                            s2.unit_vector,
                            s3.unit_vector,
                            SENSOR_VARIANCE
                        )
                        triangle_catalogue.append(
                            CatalogueTriangle(
                                s1.id, s2.id, s3.id, triangle.A, triangle.J))
            # i += 1
        classify_bar.finish()
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalogue)))
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

    def save_to_file(self, catalog: [CatalogueTriangle]):
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


generator = CatalogueGenerator()
generator.generate_triangles()
