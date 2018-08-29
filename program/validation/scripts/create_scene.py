import datetime
import os

import numpy as np
from progress.bar import Bar

from program.const import MAIN_PATH, MAX_MAGNITUDE, CAMERA_FOV, FOCAL_LENGTH
from program.validation.scripts.simulator import (
    EquidistantCamera,
    EquisolidAngleCamera,
    OrthographicCamera,
    RectilinearCamera,
    Scene,
    StarCatalog,
    StarDetector,
    StereographicCamera
)


def create_scene(num_scenes: int = 1000, max_magnitude: int = 6):
    # resolution
    res_x = 1920  # pixels
    res_y = 1440  # pixels

    # normalized focal length
    f = FOCAL_LENGTH

    # pixel aspect ratio
    pixel_ar = 1

    # normalized principal point
    ppx = 0.5
    ppy = 0.5

    gaussian_noise_sigma = 20e-6  # rad

    cam = 0

    # magnitude parameters

    A_pixel = 525  # photonelectrons/s mm
    sigma_pixel = 525  # photonelectrons/s mm

    sigma_psf = 0.5  # pixel
    t_exp = 0.2  # s
    aperture = 15  # mm

    base_photons = 19100  # photoelectrons per mm² and
    # second of a magnitude 0 G2 star

    magnitude_gaussian = 0.01  # mag

    # star count

    min_true = 5
    max_true = 100
    min_false = 0
    max_false = 0
    min_stars = min_true

    catalog = StarCatalog(
        max_magnitude=max_magnitude,
        filename=os.path.join(
            MAIN_PATH, 'program/validation/data/hip_main.dat'))

    cameras = [
        RectilinearCamera,
        EquidistantCamera,
        EquisolidAngleCamera,
        StereographicCamera,
        OrthographicCamera,
    ]

    camera = cameras[cam](f, (res_x, res_y), pixel_ar, (ppx, ppy))

    detector = StarDetector(A_pixel, sigma_pixel, sigma_psf, t_exp, aperture,
                            base_photons)

    inputs = []
    outputs = []

    classify_bar = Bar(
        'Building scenes', max=num_scenes)
    for i in range(num_scenes):
        scene = Scene.random(
            catalog=catalog, camera=camera, detector=detector,
            min_true=min_true, max_true=max_true,
            min_false=min_false, max_false=max_false,
            min_stars=min_stars, max_tries=1000000,
            gaussian_noise_sigma=gaussian_noise_sigma,
            magnitude_gaussian=magnitude_gaussian)

        if not scene:
            raise Exception('No scene generated')

        # inputs.append(np.hstack(
        #     (scene.pos[::, ::-1], scene.magnitudes.reshape(-1, 1))).flatten())
        inputs.append(np.hstack((
            scene.magnitudes.reshape(-1, 1),
            scene.uv[::, ::])).flatten())
        outputs.append(scene.ids)
        classify_bar.next()

    classify_bar.finish()

    def write_csv(filename, lines):
        with open(filename, 'w') as f:
            for line in lines:
                f.write(','.join(str(value) for value in line) + '\n')

    now = datetime.datetime.now()
    write_csv(os.path.join(
        MAIN_PATH,
        'tests/scenes/input_sample_'
        '{}_{}_{}_{}_{}.csv'.format(
            now.year, now.month, now.day, now.hour, now.minute)), inputs)
    write_csv(os.path.join(
        MAIN_PATH,
        'tests/scenes/result_sample_'
        '{}_{}_{}_{}_{}.csv'.format(
            now.year, now.month, now.day, now.hour, now.minute)), outputs)


create_scene(num_scenes=1000, max_magnitude=5)
