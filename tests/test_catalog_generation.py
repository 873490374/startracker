import csv
import os

import pytest

from program.catalog.catalog_generator import CatalogGenerator
from program.const import MAIN_PATH, SENSOR_VARIANCE, CAMERA_FOV
from program.planar_triangle import ImagePlanarTriangle
from program.star import StarUV


class TestCatalogGeneration:
    def test_catalog_generation(self):
        generator = CatalogGenerator(
            max_magnitude=1, sensor_variance=SENSOR_VARIANCE,
            camera_fov=CAMERA_FOV)
        catalog = generator.generate_triangles(
            os.path.join(MAIN_PATH, 'program/validation/data/hip_main.dat'))
        cat = []
        with open(os.path.join(
                MAIN_PATH, 'tests/catalog/catalog_generation_test_data.csv'),
                'r', newline='') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                cat.append(ImagePlanarTriangle(
                    StarUV(int(row['star1_id']), None, None),
                    StarUV(int(row['star2_id']), None, None),
                    StarUV(int(row['star3_id']), None, None),
                    float(row['area']), float(row['moment']),
                    None, None))
        assert len(catalog) == 364
        assert len(catalog) == len(cat)
        i = 0
        for i in range(len(catalog)):
            t1 = catalog[i]
            t2 = cat[i]
            assert t1.s1.id == t2.s1.id
            assert t1.s2.id == t2.s2.id
            assert t1.s3.id == t2.s3.id
            assert t1.area == t2.area
            assert t1.moment == t2.moment
        # generator.save_to_file(catalog, os.path.join(
        #     MAIN_PATH, 'tests/catalog/catalog_generation_test_data.csv'))

    @pytest.mark.skip('Not ready')
    def test_save_catalog(self):
        pass
