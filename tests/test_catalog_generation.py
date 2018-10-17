import datetime

import numpy as np
import os


from program.parallel.triangle_catalog_generator_parallel import (
    TriangleCatalogGeneratorParallel,
)
from program.const import MAIN_PATH


class TestCatalogGeneration:
    def test_catalog_generation(self):
        star_catalog_fp = os.path.join(
            MAIN_PATH, 'program/validation/data/hip_main.dat')
        expected_data_fp = os.path.join(
            MAIN_PATH, 'tests/catalog/catalog_generation_test_data.csv')
        generator = TriangleCatalogGeneratorParallel(max_magnitude=2)
        catalog, m, q = generator.generate_triangles(star_catalog_fp)
        with open(expected_data_fp, 'rb') as f:
            cat = np.genfromtxt(f, dtype=np.float64, delimiter=',')
        assert np.isclose(0.00225361039331, m)
        assert np.isclose(-0.00225062563389, q)
        assert len(catalog) == 5
        assert len(catalog) == len(cat)
        for i in range(len(catalog)):
            t1 = catalog[i]
            t2 = cat[i]
            assert t1[0] < t1[1]
            assert t1[1] < t1[2]
            assert t1[0] == t2[0]
            assert t1[1] == t2[1]
            assert t1[2] == t2[2]
            assert t1[3] == t2[3]
            assert t1[4] == t2[4]

        for i in range(1, 50):
            timestamp = datetime.datetime.now()
            test_fp = os.path.join(
                MAIN_PATH, './program/catalog/generated/triangle_catalog_partial_{}_{}_{}.{}'.format(
                    timestamp.year, timestamp.month, timestamp.day, i))
            os.remove(test_fp)
