import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import scipy
from scipy import ndimage


class BlurFilters:

    def __init__(self):
        pass

    def sharpen_imge(self, filename, value):
        # img = Image.open(filename).convert('RGB')
        # img = np.asarray(img)
        img = cv2.imread(filename)


        if value == 0:
            return img
        if value == 1:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            img2 = cv2.filter2D(img, -1, kernel)
            return cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        elif value == 2:
            kernel = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]])
            img2 = cv2.filter2D(img, -1, kernel)
            return cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        elif value == 3:
            kernel = np.array([[-2, -2, -2], [-2, 32, -2], [-2, -2, -2]])
            img2 = cv2.filter2D(img, -1, kernel)
        return img2

    def motion_blur(self, filename, scale=20):
        img = Image.open(filename).convert('RGB')
        img = np.asarray(img)
        if scale == 0:
            return img

        size = int(img.shape[1] * (scale/100))

        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        output = cv2.filter2D(img, -1, kernel_motion_blur)
        return output

    def increase_saturation(self, filename, scale):
        img = Image.open(filename).convert('RGB')
        converter = ImageEnhance.Color(img)
        img2 = converter.enhance(scale)
        img2 = np.asarray(img2)
        return img2
