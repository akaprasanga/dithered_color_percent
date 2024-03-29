import cv2
import numpy as np
from segmentation import Segment
import time
from PIL import Image
import quantize
import os
import hitherdither
import PIL
from os import walk

seg = Segment()
# area = 0
# area_r = 0


def dither(filename, number_of_color):
    img = Image.open(filename).convert('RGB')
    palette = hitherdither.palette.Palette.create_by_median_cut(img, n=number_of_color)
    img_dithered = hitherdither.ordered.bayer.bayer_dithering(
        img, palette, 10, order=8)
    name = filename.split('/')
    name = name[len(name) - 1]
    name = 'c-'+str(number_of_color)+name

    if not os.path.exists('A_DITHERED'):
        os.makedirs('A_DITHERED')
    os_path = os.path.dirname(os.path.abspath(__file__))
    os_path =os_path.replace(os.sep, '/')
    path = os_path + '/A_DITHERED/' + name
    # print(path)
    # print(f_name,'\n', file_extension)

    img_dithered.save(path)
    return path


def scale_up_image(img, factor):
    h, w = img.shape[:2]
    h = h * factor
    w = w * factor
    img = cv2.resize(img, (w,h))
    return img

def scale_down_image(img, factor):
    h, w = img.shape[:2]
    h = h // factor
    w = w // factor
    img = cv2.resize(img, (w,h))
    return img

def get_cdn_in_segment(img):
    distribution_dict = {}
    image = Image.fromarray(img.astype('uint8'), 'RGB')
    colors = image.getcolors()

    black_pixels_number, rgb_value = colors[len(colors)-1]
    # print(colors)
    area = img.shape[0] * img.shape[1]
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


def get_max_value_from_dict(dictionary):
    values = dictionary.values()
    max_value = max(values)
    return max_value


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
        max_value = get_max_value_from_dict(color_dist)
        # if max_value > 0.8:
        #     print('passs')
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
    segmented, boundaries, segmented_r = seg.slic_superpixel(filename, segment_number,connectivity,sigma,compactness, color_pockets)
    # segmented = seg.fz_superpixel(filename, 200)
    # segmented = seg.qc_superpixel(filename,1, 20)
    original = cv2.imread(filename)
    original_r = scale_up_image(original, 3)

    original_copy = original.copy()
    original_copy_r = original_r.copy()
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_r = cv2.cvtColor(original_r, cv2.COLOR_BGR2RGB)

    # global area, area_r
    # area = original.shape[0]*original.shape[1]
    # area_r = original_r.shape[0]*original_r.shape[1]

    color_replaced = count_proportion(segmented, original,color_pockets)
    color_replaced_r = count_proportion(segmented_r, original_r,color_pockets)
    color_replaced_r = scale_down_image(color_replaced_r, 3)

    name = filename.split('/')
    name = name[len(name)-1]
    if not os.path.exists('A_OUTPUT'):
        os.makedirs('A_OUTPUT')
    cv2.imwrite('A_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name, color_replaced)
    segmented_image = cv2.imread('A_SEGMENTED/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets) + name)
    dithered_img_name = 'A_DITHERED/' + name
    dithered_img = cv2.imread(dithered_img_name)

    try:
        joined_img = np.hstack((original_copy,dithered_img,color_replaced))
        if not os.path.exists('A_STICHED_OUTPUT'):
            os.makedirs('A_STICHED_OUTPUT')
        cv2.imwrite('A_STICHED_OUTPUT/s' + str(segment_number) + '-' + str(sigma) + str(connectivity) + str(compactness) + 'c' + str(color_pockets) + name, joined_img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_replaced, 'No Resize', (0, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(color_replaced_r, 'Resized', (0, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        joined_img_r = np.hstack((original_copy, color_replaced, color_replaced_r))
        if not os.path.exists('A_RESIZED_OUTPUT'):
            os.makedirs('A_RESIZED_OUTPUT')
        cv2.imwrite('A_RESIZED_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name, joined_img_r)

    except:
        joined_img = np.hstack((original_copy,segmented_image,color_replaced))
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(color_replaced, 'No Resize', (0, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(color_replaced_r, 'Resized', (0, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        joined_img_r = np.hstack((original_copy, color_replaced, color_replaced_r))
        if not os.path.exists('A_STICHED_OUTPUT'):
            os.makedirs('A_STICHED_OUTPUT')
        if not os.path.exists('A_RESIZED_OUTPUT'):
            os.makedirs('A_RESIZED_OUTPUT')
        cv2.imwrite('A_STICHED_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name, joined_img)
        cv2.imwrite('A_RESIZED_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name, joined_img_r)
    print("Time Elapsed = ",time.time()-start)

    return color_replaced,segmented_image, time.time()-start

# f = []
# for (dirpath, dirnames, filenames) in walk('test'):
#     f.extend(filenames)
#     break
#
# counter = 0
# for each in f:
#     name = each
#     main(name,100,False,3,3,3)
#     print("processing image", counter)
#     counter += 1

# a,b,c = main('E:/Work/dithered_color_percent/images/floral.processed.png',100,False,3,3,3)
# print(a,b)
# path = dither('C:/Users/prasa/Downloads/Bishal/dithered_color_percent/images/Siam-Red-Pride.jpg',4)
# print(path)