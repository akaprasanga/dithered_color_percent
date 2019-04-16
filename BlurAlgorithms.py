import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
from PIL.ImageEnhance import Sharpness


class BlurFilters:

    def __init__(self):
        pass

    def motion_blur(self, filename, scale=20):
        # img = Image.open_filedialog(filename).convert('RGB')
        # img = np.asarray(img)
        img = filename
        if scale <= 2:
            return img

        size = int(img.shape[1] * (scale/100))

        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        output = cv2.filter2D(img, -1, kernel_motion_blur)
        return output

    def increase_saturation(self, filename, scale):
        # img = Image.open_filedialog(filename).convert('RGB')
        img = filename
        img = Image.fromarray(img.astype('uint8'))
        converter = ImageEnhance.Color(img)
        img2 = converter.enhance(scale)
        img2 = np.asarray(img2)
        return img2

    def sharpening_filter(self, img, value):
        img = Image.fromarray(img.astype('uint8'))
        return np.asarray(Sharpness(img).enhance(value), dtype='uint8')
