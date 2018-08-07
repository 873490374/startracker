import os

import numpy as np
import pytest

from program.const import MAIN_PATH, CAMERA_FOV, FOCAL_LENGTH
from program.star import StarUV, StarPosition
from program.tracker.planar_triangle_calculator import PlanarTriangleCalculator
from program.utils import read_scene, calc_vector, convert_star_to_uv
from program.validation.scripts.simulator import (
    angles_to_vector,
    vector_to_angles,
    RectilinearCamera)


class TestUtils:

    @pytest.mark.skip('Old test')
    def test_old_read_scene(self):
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'scene_read_test')
        assert len(input_data) == 1
        assert len(result) == 1

        valid_input_data = np.array([
            StarUV(-1, 1.899234433685227,
                   np.array([0.34191118, 0.93017646, -0.13367307])),
            StarUV(-1, 2.1408339115513795,
                   np.array([0.49165827, 0.55138579, 0.67397764])),
            StarUV(-1, 2.9690640538571267,
                   np.array([0.85537044, -0.00486044, 0.517994])),
            StarUV(-1, 1.9298043460377514,
                   np.array([0.01084465, -0.99385139, -0.11018987])),
            StarUV(-1, 2.7608854416083433,
                   np.array([-0.32097292, 0.90513446, -0.27876152])),
            StarUV(-1, 2.569904849798762,
                   np.array([0.92571613, -0.11994349, -0.35869655])),
            StarUV(-1, 1.6483784281102398,
                   np.array([0.03385701, -0.70661934, -0.70678343])),
            StarUV(-1, 3.001700957650252,
                   np.array([0.08804479, -0.40541719, -0.90988187])),
            StarUV(-1, 2.4727375359373367,
                   np.array([-0.00482872, -0.03424387, 0.99940184])),
            StarUV(-1, 2.4813444701656047,
                   np.array([0.01163716, -0.01209152, 0.99985918])),
            StarUV(-1, 2.2082238181410747,
                   np.array([0.46999456, -0.48154559, -0.73974249])),
            StarUV(-1, 2.8808625636857403,
                   np.array(
                       [-3.04004076e-04, 1.38796274e-01, -9.90320909e-01])),
            StarUV(-1, 1.9795735855128245,
                   np.array([0.73373499, 0.43550017, -0.5215099])),
            StarUV(-1, 1.8644780398686789,
                   np.array([0.80803054, -0.43016093, 0.40255213])),
        ])
        assert np.array_equal(input_data[0], valid_input_data)
        assert np.array_equal(result[0], np.array([
            -1, -1, -1, 42913, -1, -1, -1, -1, 45941, -1, 45556, -1, -1, 41037
        ]))

    def test_old_read_scene(self):
        input_data, result = read_scene(
            os.path.join(MAIN_PATH, 'tests/scenes'), 'scene_read_test')
        assert len(input_data) == 1
        assert len(result) == 1

        valid_input_data = np.array([
            StarUV(-1, 2.733499070252604,
                   np.array([
                       -0.06617460776582357,
                       0.017131632525640077,
                       0.9976609787167412])),
            StarUV(-1, 1.6780027326655544,
                   np.array([
                       -0.07210616562715774,
                       -0.044198397082433914,
                       0.9964171829981118])),
            StarUV(-1, 0.1795405131255828,
                   np.array([
                       0.06766401198764317,
                       0.027950774023394447,
                       0.9973165674514936])),
            StarUV(-1, 2.2594342986450835,
                   np.array([
                       0.025243305939648634,
                       -0.06148329729444962,
                       0.9977888452268044])),
            StarUV(-1, 1.7227576298769856,
                   np.array([
                       0.07408587116028655,
                       0.004644802910352076,
                       0.9972410488444333])),
            StarUV(-1, 2.7663874553627847,
                   np.array([
                       0.08325343217626442,
                       -0.017150049027561166,
                       0.9963808216988242])),
        ])
        assert np.array_equal(input_data[0], valid_input_data)
        assert np.array_equal(result[0], np.array([
            26241, 26311, 24436, 25930, 26727, 23875
        ]))

    @pytest.mark.skip('Don\' know if I work correctly')
    def test_calc_vector(self):
        scene_pos = [
            [1291.89879376, 1614.76960933],
            [956.5553143, 242.53855808],
            [494.001191, 1239.90900667],
            [1358.30016728, 607.74055388],
            [781.30824329, 1700.00200145],
            [1155.04964229, 1221.74476233],
        ]

        scene_ids = [45941, 45238, 41037, 48002, 42913, 45556]

        uv = [
            [-0.05130012, - 0.04141976, 0.99782398],
            [-0.13467235, 0.01115744, 0.99082737],
            [0.13052839, 0.04303672, 0.99051006],
            [-0.047868, 0.07956208, 0.99567993],
            [-0.11875961, 0.10372882, 0.98748999],
            [0.06412848, 0.11620191, 0.99115319],
        ]

        focal_length = 0.5 / np.tan(np.deg2rad(CAMERA_FOV) / 2)
        for i in range(len(scene_ids)):
            s = calc_vector(
                scene_pos[i][1],
                scene_pos[i][0],
                pixel_size=0.000525,
                focal_length=focal_length
            )
            c = uv[i]
            assert s == uv

    @pytest.mark.skip('Don\' know if I work correctly')
    def test_ddd(self):
        RAdeg = 0.31645128
        DEdeg = 19.74131648

        vec = angles_to_vector(np.deg2rad(RAdeg), np.deg2rad(DEdeg))

        angels = np.rad2deg(vector_to_angles(vec))
        assert angels[0] == RAdeg
        assert angels[1] == DEdeg

        f = FOCAL_LENGTH
        res_x = 1920  # pixels
        res_y = 1440  # pixels
        pixel_ar = 1
        # normalized principal point
        ppx = 0.5
        ppy = 0.5

        camera = RectilinearCamera(f, (res_x, res_y), pixel_ar, (ppx, ppy))
        d = np.array([np.deg2rad(RAdeg)])
        e = np.array([np.deg2rad(DEdeg)])
        yx = camera.from_angles(d, e)
        s = calc_vector(
            yx[0][0],
            yx[0][1],
            pixel_size=1,
            focal_length=np.rad2deg(f)
        )
        fov = 12.45
        focal = 0.23
        haha = 2 * np.power(np.tan(1 / (2 * focal)), -1)
        sss = 2 * np.power(np.tan(1 / (2 * f)), -1)
        print('d')
        print(vec)
        print(s)
        # trian_calc = PlanarTriangleCalculator(1)
        # print(trian_calc)

    # @pytest.mark.skip('I don\'t know how to translate xy position back to uv')
    def test_adada(self):
        uv = [
            [0.06866535, -0.0498963, 0.9963912],
            [0.07474301, 0.01140279, 0.99713763],
            [-0.06520745, -0.06038494, 0.99604299],
            [-0.02255166, 0.02891264, 0.99932751],
            [-0.07155564, -0.03705379, 0.99674812],
            [-0.08065246, -0.01520595, 0.99662629],
        ]

        orientation = np.array([
            [0.81230405, -0.16834913, -0.55840908],
            [-0.56588527, 0.00429001, -0.82447283],
            [0.14119486, 0.9857181, -0.0917815],
        ])

        pos = [
            [312.08786619, 1747.73090906],
            [54.77276898, 1678.35409499],
            [845.4802719, 137.50184802],
            [1037.46751659, 1207.62253161],
            [170.51234912, 203.81648496],
            [552.58255956, 1847.98312942],
        ]

        ids = [26311, 25930, 24436, 26241, 23875, 26727]

        magnitudes = [
            1.68826354, 2.26076944, 0.19990068,
            2.74317044, 2.77568215, 1.74322264,
        ]

        original_ids = [23875, 24436, 25930, 26241, 26311, 26727]
        original_deg = [
            [76.96264146, -5.08626282],
            [78.63446353, -8.20163919],
            [83.00166562, -0.2990934],
            [83.85825475, -5.90989984],
            [84.05338572, -1.20191725],
            [85.18968672, -1.94257841],
        ]
        calc_uv = []
        for deg in original_deg:
            calc_uv.append(
                convert_star_to_uv(StarPosition(-1, 1, deg[0], deg[1])))
        ca_uv = [s.unit_vector for s in calc_uv]
        ca_uv_orient = np.dot(ca_uv, orientation.transpose())
        for i in range(len(uv)):
            for j in range(3):
                assert np.isclose(ca_uv_orient[i][j], uv[i][j])

        # This does not work properly
        # for i in range(len(uv)):
        #     a = calc_vector(pos[i][1], pos[i][0], 1, FOCAL_LENGTH)
        #     b = ca_uv_orient[i]
        #     # print(a)
        #     # print(b)

        trian_calc = PlanarTriangleCalculator(1)
        t1 = trian_calc.calculate_triangle(
            StarUV(-1, -1, ca_uv_orient[0]),
            StarUV(-1, -1, ca_uv_orient[1]),
            StarUV(-1, -1, ca_uv_orient[2]))

        t2 = trian_calc.calculate_triangle(
            calc_uv[0],
            calc_uv[1],
            calc_uv[2])

        assert np.isclose(t1.area, t2.area)
        assert np.isclose(t1.moment, t2.moment)
        assert np.isclose(t1.area_var, t2.area_var)
        assert np.isclose(t1.moment_var, t2.moment_var)

        # This does not work properly
        # f = FOCAL_LENGTH
        # res_x = 1920  # pixels
        # res_y = 1440  # pixels
        # pixel_ar = 1
        # # normalized principal point
        # ppx = 0.5
        # ppy = 0.5
        #
        # camera = RectilinearCamera(f, (res_x, res_y), pixel_ar, (ppx, ppy))
        # angles = camera.to_angles(np.array(pos))
        # print(angles)
