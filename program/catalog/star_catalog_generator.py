import datetime
import numpy as np
import os

from program.const import MAIN_PATH
from program.utils import convert_star_to_uv
from program.validation.scripts.simulator import StarCatalog


class StarCatalogGenerator:
    def __init__(
            self, max_magnitude: float):
        self.max_magnitude = max_magnitude

    def generate_stars(self, star_catalog_path: str) -> np.ndarray:
        converted_stars = []

        stars = self.read_catalogue_stars(star_catalog_path)
        for s in stars:
            if s[1] <= self.max_magnitude:
                star = convert_star_to_uv(s)
                converted_stars.append(
                    np.array([s[0], star[0], star[1], star[2]]))
        converted_stars = np.array(converted_stars)
        return converted_stars

    def read_catalogue_stars(self, star_catalog_path: str) -> [float]:
        stars = []

        catalog = StarCatalog(self.max_magnitude, star_catalog_path).catalog
        for row in catalog.itertuples():
            s = np.array([
                row[2],
                row[6],
                row[9],
                row[10],
            ])
            stars.append(s)
        return stars

    def save_to_file(self, catalog: np.ndarray, output_file_path: str):
        np.savetxt(output_file_path, catalog, delimiter=',')


generator = StarCatalogGenerator(6.2)
catalog = generator.generate_stars(
    os.path.join(MAIN_PATH, 'program/validation/data/hip_main.dat'))
now = datetime.datetime.now()
generator.save_to_file(catalog, os.path.join(
    MAIN_PATH, './program/catalog/generated/star_catalog_'
               '{}_{}_{}_{}_{}.csv'.format(
                now.year, now.month, now.day, now.hour, now.minute)))
