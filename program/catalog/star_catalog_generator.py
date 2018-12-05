import csv
import datetime
import operator
import os

from program.const import MAIN_PATH
from program.star import StarPosition, StarUV
from program.utils import convert_star_to_uv
from program.validation.scripts.simulator import StarCatalog


class StarCatalogGenerator:
    def __init__(
            self, max_magnitude: float):
        self.max_magnitude = max_magnitude

    def generate_stars(
            self, star_catalog_path: str) -> []:
        converted_stars = []

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s.magnitude <= self.max_magnitude:
                star = convert_star_to_uv(s)
                converted_stars.append(star)
        converted_stars = self.sort_catalog(converted_stars)
        return converted_stars

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
        return sorted(catalog, key=operator.attrgetter('id'))

    def save_to_file(
            self, catalog: [StarUV], output_file_path: str):

        with open(output_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=[
                'id', 'magnitude', 'uv_i', 'uv_j', 'uv_k'])
            csvwriter.writeheader()
            for t in catalog:
                csvwriter.writerow(dict(t))


generator = StarCatalogGenerator(6.2)
catalog = generator.generate_stars(
    os.path.join(MAIN_PATH, 'program/validation/data/hip_main.dat'))
now = datetime.datetime.now()
generator.save_to_file(catalog, os.path.join(
    MAIN_PATH, './program/catalog/generated/star_catalog_'
               '{}_{}_{}_{}_{}.csv'.format(
                now.year, now.month, now.day, now.hour, now.minute)))
