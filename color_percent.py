import cv2
import numpy as np
from segmentation import Segment
import matplotlib.pyplot as plt
import time
from PIL import Image

seg = Segment()
area = 0


def quantize_color(color_dict):
    num_colors = len(color_dict)
    quantized_color = {}
    key_list = []
    value_list = []
    np_values = np.array(value_list)
    quantize_bin = np.array([0, 0.25, 0.5, 0.75, 1])
    for k,v in color_dict.items():
        key_list.append(k)
        value_list.append(v)
    if len(key_list) == 1:
        quantize_color[key_list[0]] = value_list[0]
        return quantized_color
    if len(key_list) == 2:
        digitized = np.digitize(np_values, quantize_bin)


def join_images(fname, segment_number, color_replaced):
    path = fname.split('/')
    fname = path[len(path)-1]
    fname ='segmented/slic' + str(segment_number) +fname
    img = cv2.imread(fname)
    joined = np.hstack((img, color_replaced))
    return joined


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
    image = convert_to_pillow_format(img)
    print(image)
    colors = image.getcolors()
    print(colors)
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
        print(color_dist)
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
    # cv2.imwrite('rgb.jpg', final1)
    # cv2.imwrite('bgr.jpg', final2)
    return final2


filename = 'Ascenure.processed.png'
filename = 'images/' + filename


start = time.time()
segment_number = 50
segmented, boundaries = seg.slic_superpixel(filename, segment_number)
# segmented = seg.fz_superpixel(filename, 200)
# segmented = seg.qc_superpixel(filename,1, 20)
original = cv2.imread(filename)
print(original.shape)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
get_all_colors_of_image(original)

area = original.shape[0]*original.shape[1]
color_replaced = count_proportion(segmented, original)
joined = join_images(filename, segment_number, color_replaced)
cv2.imwrite(str(segment_number)+'slic.jpg', joined)
print(time.time()-start)
