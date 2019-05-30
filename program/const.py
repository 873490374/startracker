import numpy as np
import os

MAX_MAGNITUDE = 5
CAMERA_FOV = 10  # angular distance in degrees
COS_CAMERA_FOV = np.cos(np.deg2rad(CAMERA_FOV))
# noinspection LongLine
FOCAL_LENGTH_NORM = 0.5 / np.tan(np.deg2rad(CAMERA_FOV) / 2)  # normalized,in pixels
SENSOR_VARIANCE = 270e-6 / 10
SIG_X = 3
MAIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
TRACKING_MODE_ENABLED = False
