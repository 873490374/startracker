import numpy as np
import os

MAX_MAGNITUDE = 3
CAMERA_FOV = 10  # angular distance in degrees
FOCAL_LENGTH = 0.5 / np.tan(np.deg2rad(10) / 2)  # in pixels
SENSOR_VARIANCE = 8  # np.pow(87.2665e-6/3, 2)
MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
