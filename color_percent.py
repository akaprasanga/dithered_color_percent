import cv2
import numpy as np
from segmentation import Segment
import matplotlib.pyplot as plt
import time
from PIL import Image
import quantize
from os import walk

seg = Segment()
area = 0


def join_images(fname, segment_number, color_replaced):
    path = fname.split('/')
    fname = path[len(path)-1]
    dithered_name = fname.split('.')[0]
    original_name = 'originalImages/' +dithered_name + '.png'

    dithered_name = '6colordither/'+dithered_name+'.processed.png'

    fname ='segmented/slic' + str(segment_number) +fname
    segmented_img = cv2.imread(fname)
    dithered_img = cv2.imread(dithered_name)
    original_img = cv2.imread(original_name)
    print(original_name, dithered_name)
    print(original_img.shape, dithered_img.shape, color_replaced.shape)
    final = np.hstack((original_img, dithered_img, color_replaced))
    # joined = np.hstack((segmented_img, color_replaced))
    # final = np.vstack((joined1, joined))
    return final


def get_cdn_in_segment(img):
    distribution_dict = {}
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    colors = image.getcolors()
    # print(colors)
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
        masked_segment = cv2.cvtColor(masked_segment,cv2.COLOR_BGR2RGB)
        color_dist = get_cdn_in_segment(masked_segment)
        # print(segVal, color_dist)
        # print(color_dist)
        cmy_color = quantize.rgb_to_cmy(color_dist)
        q_color = quantize.quantize_color(cmy_color)
        # q_color = cmy_color

        mean_c = 0
        mean_m = 0
        mean_y = 0
        # print(color_dist)
        for (key, value) in q_color.items():
            c, m, y = key
            mean_c = mean_c + c * value
            mean_m = mean_m + m * value
            mean_y = mean_y + y * value

        mean_r = quantize.cmy_to_rgb(mean_c)
        mean_g = quantize.cmy_to_rgb(mean_m)
        mean_b = quantize.cmy_to_rgb(mean_y)

        mask_r[segments == segVal] = int(mean_r)  # Five start code for replacing values
        mask_g[segments == segVal] = int(mean_g)  # Five start code for replacing values
        mask_b[segments == segVal] = int(mean_b)  # Five start code for replacing values

        # print(mean_r, mean_g, mean_b)

    final1 = np.dstack((mask_r, mask_g, mask_b))
    final2 = np.dstack((mask_b, mask_g, mask_r))
    # cv2.imwrite('rgb.jpg', final1)
    # cv2.imwrite('bgr.jpg', final2)
    return final1


def main(filename):
    # filename = 'IMG_20180612_0002_Page_1.processed.png'
    name = filename
    filename = '6colordither/' + filename


    start = time.time()
    segment_number = 100
    segmented, boundaries = seg.slic_superpixel(filename, segment_number)
    # segmented = seg.fz_superpixel(filename, 200)
    # segmented = seg.qc_superpixel(filename,1, 20)
    original = cv2.imread(filename)
    print(original.shape)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    get_all_colors_of_image(original)

    global area
    area = original.shape[0]*original.shape[1]
    color_replaced = count_proportion(segmented, original)
    joined = join_images(filename, segment_number, color_replaced)
    # joined = cv2.cvtColor(joined, cv2.COLOR_RGB2BGR)

    cv2.imwrite('quant4level/'+str(segment_number)+name, joined)
    print(time.time()-start)

f = []
for (dirpath, dirnames, filenames) in walk('6colordither'):
    f.extend(filenames)
    break

counter = 0
for each in f:
    name = each
    main(name)
    print("processing image", counter)
    counter += 1

# main('Ascenure.processed.png')