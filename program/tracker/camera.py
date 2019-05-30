import cv2

from PIL import Image


class CameraConnector:

    @staticmethod
    def get_image() -> Image.Image:
        width = 900  # pixels
        height = 900  # pixels
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
        return image

#
# c = CameraConnector()
# img = c.get_image()
# print(img)
