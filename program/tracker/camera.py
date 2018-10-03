import cv2

from PIL import Image


class CameraConnector:

    def get_image(self) -> Image.Image:
        width = 1920  # pixels
        height = 1440  # pixels
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
        return image

#
# c = CameraConnector()
# img = c.get_image()
# print(img)
