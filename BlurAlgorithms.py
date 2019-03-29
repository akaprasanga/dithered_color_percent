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
        img = Image.open(filename).convert('RGB')
        img = np.asarray(img)
        # f = scipy.misc.face(gray=True).astype(float)

        blurred_f = ndimage.gaussian_filter(img, 3)


        filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)

        alpha = 30
        sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
        return sharpened

        # if value == 0:
        #     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if value == 1:
        #     kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        #     img2 = cv2.filter2D(img, -1, kernel)
        #     return cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # elif value == 2:
        #     kernel = np.array([[-2, -2, -2], [-2, 17, -2], [-2, -2, -2]])
        #     img2 = cv2.filter2D(img, -1, kernel)
        #     return cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # elif value == 3:
        #     kernel = np.array([[-2, -2, -2], [-2, 32, -2], [-2, -2, -2]])
        #     img2 = cv2.filter2D(img, -1, kernel)
        #     return cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    def motion_blur(self, filename, scale= 20):
        img = cv2.imread(filename)
        if scale == 2:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size = int(img.shape[1] * (scale/100))

        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        output = cv2.filter2D(img, -1, kernel_motion_blur)
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    def increase_saturation(self, filename, scale):
        img = Image.open(filename).convert('RGB')
        converter = ImageEnhance.Color(img)
        img2 = converter.enhance(scale)
        img2 = np.asarray(img2)
        return img2
