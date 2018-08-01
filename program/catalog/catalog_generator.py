import csv

import numpy as np
from progress.bar import Bar

from program.planar_triangle import CatalogPlanarTriangle
from program.star import StarPosition, StarUV
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.tracker.star_identifier import StarIdentifier
from program.validation.scripts.simulator import StarCatalog


class CatalogGenerator:
    def __init__(self, max_magnitude, sensor_variance, camera_fov):
        self.star_identifier = StarIdentifier(
            planar_triangle_calculator=PlanarTriangleCalculator(
                sensor_variance=sensor_variance
            ),
            sensor_variance=sensor_variance,
            max_magnitude=max_magnitude,
            camera_fov=camera_fov,
            catalog=None)
        self.triangle_calc = PlanarTriangleCalculator(
            sensor_variance=sensor_variance)
        self.max_magnitude = max_magnitude
        self.sensor_variance = sensor_variance
        self.camera_fov = camera_fov

    def generate_triangles(
            self, star_catalog_path: str) -> [CatalogPlanarTriangle]:
        converted_start = []
        triangle_catalogue = []

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s.magnitude <= self.max_magnitude:
                star = self.convert(s)
                converted_start.append(star)
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
                            s1, s2, s3, self.max_magnitude, self.camera_fov):
                        triangle = self.triangle_calc.calculate_triangle(
                            s1, s2, s3)
                        triangle_catalogue.append(triangle)
        classify_bar.finish()
        print('Number of planar triangles in catalogue: {}'.format(
            len(triangle_catalogue)))
        # kvector
        # self.save_to_file(triangle_catalogue)
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

    def save_to_file(
            self, catalog: [CatalogPlanarTriangle], output_file_path: str):

        with open(output_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=[
                'star1_id', 'star2_id', 'star3_id', 'area', 'moment'])
            csvwriter.writeheader()
            for t in catalog:
                csvwriter.writerow(dict(t))
