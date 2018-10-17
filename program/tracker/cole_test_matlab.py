import os

import numpy as np

from program.const import MAIN_PATH
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator

SENSOR_VARIANCE = 0.001
sig_x = 3
filename = os.path.join(
    MAIN_PATH, 'tests/catalog/{}.csv'.format(
        'triangle_catalog_mag6_fov10_full_area'))
with open(filename, 'rb') as f:
    catalog = np.genfromtxt(f, dtype=np.float64, delimiter=',')

kv_m = 1.12594254671E-08
kv_q = -2.39815865128E-09

planar_triangle_calc = PlanarTriangleCalculator(
    sensor_variance=SENSOR_VARIANCE
)
kvector_calculator = KVectorCalculator(kv_m, kv_q)

star_1_id = 2484
star_1_deg = (7.88566992, -62.95808549)
star_1_uv = [0.45034301,  0.06237546, -0.89067417]
star_2_id = 2487
star_2_deg = (7.88898014, -62.96544985)
star_2_uv = [0.450226,  0.06238576, -0.8907326]
star_3_id = 2578
star_3_deg = (8.18247614, -63.03137868)
star_3_uv = [0.44888572,  0.06454528, -0.89125502]
a = 1.34110725693212E-07
m = 5.18533679165969E-14
k = 5

t = planar_triangle_calc.calculate_triangle(
    np.array([star_1_id, star_1_uv[0], star_1_uv[1], star_1_uv[2]]),
    np.array([star_2_id, star_2_uv[0], star_2_uv[1], star_2_uv[2]]),
    np.array([star_3_id, star_3_uv[0], star_3_uv[1], star_3_uv[2]]))

area = t[3]
moment = t[4]
area_var = t[5]
moment_var = t[6]

area_min = area - sig_x * area_var
area_max = area + sig_x * area_var

k_min, k_max = kvector_calculator.find_in_kvector(area_min, area_max, catalog)
k_range = k_max - k_min

print(k_range)
