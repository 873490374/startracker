import datetime
import os

from program.parallel.triangle_catalog_generator_parallel import (
    TriangleCatalogGeneratorParallel,
)
from program.const import MAIN_PATH

# magnitude = 5.8
generator = TriangleCatalogGeneratorParallel(5)
catalog, m, q = generator.generate_triangles(
    os.path.join(MAIN_PATH, 'program/validation/data/hip_main.dat'))
now = datetime.datetime.now()
generator.save_to_file(catalog, os.path.join(
    MAIN_PATH, './program/catalog/generated/triangle_catalog_full_'
               '{}_{}_{}_{}_{}.csv'.format(
                now.year, now.month, now.day, now.hour, now.minute)))
generator.save_m_q_to_file(m, q, os.path.join(
    MAIN_PATH, './program/catalog/generated/triangle_catalog_m_q_'
               '{}_{}_{}_{}_{}.csv'.format(
                now.year, now.month, now.day, now.hour, now.minute)))