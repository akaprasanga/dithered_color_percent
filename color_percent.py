import cv2
import numpy as np
from segmentation import Segment
import time
from PIL import Image
import quantize
import os
import hitherdither
import PIL


seg = Segment()
area = 0


def dither(filename, number_of_color):
    img = Image.open(filename).convert('RGB')
    palette = hitherdither.palette.Palette.create_by_median_cut(img, n=number_of_color)
    img_dithered = hitherdither.ordered.bayer.bayer_dithering(
        img, palette, 10, order=8)
    name = filename.split('/')
    name = name[len(name) - 1]

    if not os.path.exists('dithered'):
        os.makedirs('dithered')
    os_path = os.path.dirname(os.path.abspath(__file__))
    os_path =os_path.replace(os.sep, '/')
    path = os_path + '/dithered/' + name
    print(path)
    f_name, file_extension = os.path.splitext(path)
    print(f_name,'\n', file_extension)

    img_dithered.save(path)
    return path


def get_cdn_in_segment(img):
    distribution_dict = {}
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    colors = image.getcolors()

    black_pixels_number, rgb_value = colors[len(colors)-1]
    # print(colors)
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
    colors = image.getcolors()
    colors = list(zip(*colors))[1]
    # print(colors)


def count_proportion(segments, image, color_pockets):
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
        q_color = quantize.quantize_into_pockets(cmy_color,pocket_number=color_pockets)
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

    return final1


def main(filename, segment_number, connectivity, compactness, sigma, color_pockets):

    start = time.time()
    segmented, boundaries = seg.slic_superpixel(filename, segment_number,connectivity,sigma,compactness, color_pockets)
    # segmented = seg.fz_superpixel(filename, 200)
    # segmented = seg.qc_superpixel(filename,1, 20)
    original = cv2.imread(filename)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    global area
    area = original.shape[0]*original.shape[1]
    color_replaced = count_proportion(segmented, original,color_pockets)
    name = filename.split('/')
    name = name[len(name)-1]
    if not os.path.exists('output'):
        os.makedirs('output')
    cv2.imwrite('output/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name, color_replaced)
    print("Time Elapsed = ",time.time()-start)
    segmented_image = cv2.imread('segmented/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets) + name)
    return color_replaced,segmented_image, time.time()-start

# f = []
# for (dirpath, dirnames, filenames) in walk('6colordither'):
#     f.extend(filenames)
#     break
#
# counter = 0
# for each in f:
#     name = each
#     main(name)
#     print("processing image", counter)
#     counter += 1

# a,b = main('4colordither/Ascenure.processed.png',100,False,3,3,3)
# print(a,b)
# path = dither('C:/Users/prasa/Downloads/Bishal/dithered_color_percent/images/Siam-Red-Pride.jpg',4)
# print(path)