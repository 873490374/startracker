import datetime
import os

from program.parallel.triangle_catalog_generator_parallel import (
    TriangleCatalogGeneratorParallel,
)
from program.const import MAX_MAGNITUDE, SENSOR_VARIANCE, CAMERA_FOV, MAIN_PATH

generator = TriangleCatalogGeneratorParallel(
    6.2, SENSOR_VARIANCE, CAMERA_FOV)
catalog = generator.generate_triangles(
    os.path.join(MAIN_PATH, 'program/validation/data/hip_main.dat'))
now = datetime.datetime.now()
generator.save_to_file(catalog, os.path.join(
    MAIN_PATH, './program/catalog/generated/triangle_catalog_'
               '{}_{}_{}_{}_{}.csv'.format(
                now.year, now.month, now.day, now.hour, now.minute)))
