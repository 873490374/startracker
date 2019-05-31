import numpy as np

import cv2


class CameraConnector:

    @staticmethod
    def get_image() -> np.ndarray:
        width = 2592  # pixels
        height = 1458  # pixels
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
        cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        _, image = cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 0)
        return image

#
# c = CameraConnector()
# while True:
#     img = c.get_image()
#     cv2.imshow('img', img)
#     if cv2.waitKey(1) == 27:
#         break
