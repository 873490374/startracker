import numpy as np
import os

MAX_MAGNITUDE = 5
CAMERA_FOV = 10  # angular distance in degrees
COS_CAMERA_FOV = np.cos(np.deg2rad(CAMERA_FOV))
FOCAL_LENGTH = 0.5 / np.tan(np.deg2rad(10) / 2)  # in pixels
SENSOR_VARIANCE = 0.3  # np.power(87.2665e-6/3, 2) or 0.1???
MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
