import cv2
import numpy as np
from segmentation import Segment
import matplotlib.pyplot as plt
import time
from PIL import Image

seg = Segment()
area = 0


def get_cdn_in_segment(img):
    distribution_dict = {}
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    colors = image.getcolors()
    black_pixels_number, rgb_value = colors[len(colors)-1]
    segment_area = area - black_pixels_number

    for i in range (0, len(colors)-1):
        pixels, c = colors[i]
        color_percent = (pixels/segment_area)
        distribution_dict[c] = color_percent

    return distribution_dict


def get_colors(img):
    c_tuple = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    return c_tuple


def convert_to_pillow_format(img):
    return Image.fromarray(img.astype('uint8'), 'RGB')


def get_all_colors_of_image(img):
    image = convert_to_pillow_format(original)
    colors = image.getcolors()
    colors = list(zip(*colors))[1]
    # print(colors)


def count_proportion(segments, image):
    orig = image
    mask_r = np.zeros(image.shape[:2], dtype="uint8")
    mask_g = np.zeros(image.shape[:2], dtype="uint8")
    mask_b = np.zeros(image.shape[:2], dtype="uint8")

    for (i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255  # Five start code for replacing values

        masked_segment = cv2.bitwise_and(orig, orig, mask=mask)
        color_dist = get_cdn_in_segment(masked_segment)
        # print(segVal, color_dist)

        mean_r = 0
        mean_g = 0
        mean_b = 0
        for (key, value) in color_dist.items():
            r, g, b = key
            mean_r = mean_r + r * value
            mean_g = mean_g + g * value
            mean_b = mean_b + b * value

        mask_r[segments == segVal] = int(mean_r)  # Five start code for replacing values
        mask_g[segments == segVal] = int(mean_g)  # Five start code for replacing values
        mask_b[segments == segVal] = int(mean_b)  # Five start code for replacing values

        # print(mean_r, mean_g, mean_b)

    final1 = np.dstack((mask_r, mask_g, mask_b))
    final2 = np.dstack((mask_b, mask_g, mask_r))
    cv2.imwrite('rgb.jpg', final1)
    cv2.imwrite('bgr.jpg', final2)




filename = 'Wagion Lanet.png'
filename = 'images/' + filename


start = time.time()
segmented = seg.slic_superpixel(filename, 10000)
original = cv2.imread(filename)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
get_all_colors_of_image(original)

area = original.shape[0]*original.shape[1]
count_proportion(segmented, original)
print(time.time()-start)
