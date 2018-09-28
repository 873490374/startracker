import math
import os

import numpy as np

from program.const import MAIN_PATH
from program.tracker.kvector_calculator import KVectorCalculator
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator

SENSOR_VARIANCE = 5
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

# TODO add RA DEG from HIP.dat
star_1_id = 42913
star_1_deg = (262.96050661, -49.87598159)
star_1_uv = [-0.07897887, -0.63958635, -0.76465132]
star_2_id = 41037
star_2_deg = (263.91504719, -46.50559076)
star_2_uv = [-0.0729601 , -0.68440588, -0.72544154]
star_3_id = 45556
star_3_deg = (269.19760682, -44.34219871)
star_3_uv = [-0.01001532, -0.71510802, -0.69894221]


# TODO add from scene_old_read_test_input.csv x, y and convert to uv
starx_1_id = 42913
starx_1_deg = (262.96050661, -49.87598159)
starx_1_uv = [-0.06610610, 0.02686359, 0.99745092]
starx_2_id = 41037
starx_2_deg = (263.91504719, -46.50559076)
starx_2_uv = [-0.00632643, 0.02950380, 0.99954465]
starx_3_id = 45556
starx_3_deg = (269.19760682, -44.34219871)
starx_3_uv = [0.05445621, 0.03784688, 0.99779864]

t = planar_triangle_calc.calculate_triangle(
    np.array([star_1_id, star_1_uv[0], star_1_uv[1], star_1_uv[2]]),
    np.array([star_2_id, star_2_uv[0], star_2_uv[1], star_2_uv[2]]),
    np.array([star_3_id, star_3_uv[0], star_3_uv[1], star_3_uv[2]]))

tx = planar_triangle_calc.calculate_triangle(
    np.array([starx_1_id, starx_1_uv[0], starx_1_uv[1], starx_1_uv[2]]),
    np.array([starx_2_id, starx_2_uv[0], starx_2_uv[1], starx_2_uv[2]]),
    np.array([starx_3_id, starx_3_uv[0], starx_3_uv[1], starx_3_uv[2]]))


area = t[3]
areax = tx[3]
moment = t[4]
momentx = tx[4]
area_var = t[5]
area_varx = tx[5]
moment_var = t[6]
moment_varx = tx[6]

area_min = area - sig_x * math.sqrt(area_var)
area_max = area + sig_x * math.sqrt(area_var)

k_min, k_max = kvector_calculator.find_in_kvector(area_min, area_max, catalog)
k_range = k_max - k_min

print(k_range)
a = []
#
# for t in catalog:
#     if (
#             t[0] == 41037 and t[1] == 42913 and t[2] == 45556 or
#             t[0] == 41037 and t[1] == 45556 and t[2] == 42913 or
#             t[0] == 42913 and t[1] == 41037 and t[2] == 45556 or
#             t[0] == 42913 and t[1] == 45556 and t[2] == 41037 or
#             t[0] == 45556 and t[1] == 42913 and t[2] == 41037 or
#             t[0] == 45556 and t[1] == 41037 and t[2] == 42913):
#         print(t)
#         a.append(t)
# [42913 41037 45556
#  5.14629033e-03 5.23847270e-06 8.91106000e+05]

# print(a)
