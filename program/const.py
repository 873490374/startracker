import numpy as np
import os

MAX_MAGNITUDE = 3
CAMERA_FOV = 10  # angular distance in degrees
FOCAL_LENGTH = 0.5 / np.tan(np.deg2rad(10) / 2)  # in pixels
SENSOR_VARIANCE = 1
MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
