import cv2
import numpy as np
from segmentation import Segment
import time
from PIL import Image
import os
from dither_from_dll import FunctionsFromDLL


dllFunctions = FunctionsFromDLL()
seg = Segment()
# area = 0
# area_r = 0

def resize_img_to_dim(img, dim):
    img = cv2.resize(img, dim)
    return img


def dither(filename, number_of_color, dir_path):
    saving_path = dllFunctions.dither_image(filename, number_of_color, dir_path)
    name = filename.split('/')
    name = name[len(name) - 1]
    return saving_path + name


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


# def count_proportion(segments, image, color_pockets):
#     pr.enable()
#
#     orig = image
#     mask_r = np.zeros(image.shape[:2], dtype="uint8")
#     mask_g = np.zeros(image.shape[:2], dtype="uint8")
#     mask_b = np.zeros(image.shape[:2], dtype="uint8")
#     zeros = np.zeros(image.shape[:2], dtype="uint8")
#
#     start_color_time = time.time()
#     for (i, segVal) in enumerate(np.unique(segments)):
#         mask = zeros.copy()
#         mask[segments == segVal] = 255  # Five start code for replacing values
#
#         masked_segment = cv2.bitwise_and(orig, orig, mask=mask)
#         masked_segment = cv2.cvtColor(masked_segment, cv2.COLOR_BGR2RGB)
#         color_dist = get_cdn_in_segment(masked_segment)
#
#         cmy_color = color_dist
#         # cmy_color = quantize.rgb_to_cmy(color_dist)
#         q_color = quantize.quantize_into_pockets(cmy_color, pocket_number=color_pockets)
#
#         mean_c = 0
#         mean_m = 0
#         mean_y = 0
#         for (key, value) in q_color.items():
#             c, m, y = key
#             mean_c = mean_c + c * value
#             mean_m = mean_m + m * value
#             mean_y = mean_y + y * value
#
#         # mean_r = quantize.cmy_to_rgb(mean_c)
#         # mean_g = quantize.cmy_to_rgb(mean_m)
#         # mean_b = quantize.cmy_to_rgb(mean_y)
#
#
#         mask_r[segments == segVal] = int(mean_c)  # Five start code for replacing values
#         mask_g[segments == segVal] = int(mean_m)  # Five start code for replacing values
#         mask_b[segments == segVal] = int(mean_y)  # Five start code for replacing values
#
#
#
#     final1 = np.dstack((mask_r, mask_g, mask_b))
#     end_color_time = time.time()
#     print('Recoloring time = ', end_color_time - start_color_time)
#     pr.disable()
#     s = StringIO.StringIO()
#     sortby = 'cumulative'
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print(s.getvalue())
#     return final1
#
#
# def check_and_reduce_color(filename):
#     ready_to_process = Image.open(filename).convert('RGB')
#     color = ready_to_process.getcolors()
#     if color == None:
#         reduced_img, cluster_number = cluster.mini_batch_kmeans(filename, 24)
#         cv2.imwrite(filename,reduced_img)


def main(filename, dither_flag, dither_color, segment_number, connectivity, compactness, sigma, color_pockets, resize_flag, resize_factor, reduce_color_number, dim_change_flag, dim, grayscale_flag):
    print('Processing started ==', filename)
    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\', '/')
    actual_name = os.path.splitext(filename)[0]
    actual_name = actual_name.split('/')
    actual_name = actual_name[len(actual_name) - 1]

    if grayscale_flag == True:
        im = cv2.imread(filename, 0)
        if not os.path.exists('A_Grayscaled'):
            os.makedirs('A_Grayscaled')
        path_after_grayscale = dir_path + '/A_Grayscaled/' + actual_name + '.png'
        im = np.dstack((im, im, im))
        cv2.imwrite(path_after_grayscale, im)
        filename = path_after_grayscale

    no_dim_change = cv2.imread(filename)
    orig_dim = tuple((no_dim_change.shape[1], no_dim_change.shape[0]))
    if (dim_change_flag == True) and (dim != orig_dim):
        dim_changed = cv2.resize(no_dim_change, dim)
        if not os.path.exists('A_900X1200'):
            os.makedirs('A_900X1200')
        path_after_dim_changed = dir_path + '/A_900X1200/' + actual_name + '.png'
        cv2.imwrite(path_after_dim_changed, dim_changed)
        filename = path_after_dim_changed

    original_without_dither = cv2.imread(filename)

    if resize_flag == True:
        non_resized = cv2.imread(filename)
        upscaled = scale_up_image(non_resized,resize_factor)
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        upscaled_pil = convert_to_pillow_format(upscaled_rgb)

        if not os.path.exists('A_UPSCALED_INPUT'):
            os.makedirs('A_UPSCALED_INPUT')
        path_after_upscaled = dir_path + '/A_UPSCALED_INPUT/' + actual_name + '.png'
        upscaled_pil.save(path_after_upscaled)
        filename = path_after_upscaled

    start = time.time()
    if dither_flag == True:
        p = filename
        try:
            path = dither(p, dither_color, dir_path)
            filename = path
        except ValueError:
            print('Dither failed due to less number of Colors in Image than defined color')
            dither_flag = False

    start_slic = time.time()
    ready_for_segment_img = Image.open(filename).convert('RGB')
    segmented_r, segmented_img_path = seg.slic_superpixel(filename, np.asarray(ready_for_segment_img), segment_number,connectivity,sigma,compactness, color_pockets, resize_flag, resize_factor,dim_change_flag,dim, grayscale_flag)
    print('Slic Time = ', time.time()-start_slic)

    color_replaced_r_path = dllFunctions.replace_mean_color(filename, segmented_img_path, dir_path)
    color_replaced_r = cv2.imread(color_replaced_r_path)

    if resize_flag == True:
        color_replaced_r = scale_down_image(color_replaced_r, resize_factor)
    name = filename.split('/')
    name = name[len(name)-1]
    if not os.path.exists('A_OUTPUT'):
        os.makedirs('A_OUTPUT')
    output_img_path = 'A_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name
    cv2.imwrite(output_img_path, color_replaced_r)
    segmented_image = cv2.imread(segmented_img_path)
    dithered_img_path = 'A_DITHERED/' + name
    dithered_img = cv2.imread(dithered_img_path)

    output_img_path = dir_path + '/' + output_img_path

    color_reduce_time2 = time.time()
    reduced_color_path = dllFunctions.reduce_color(output_img_path, reduce_color_number, dir_path)
    reduced_color_img = cv2.imread(reduced_color_path)
    print('Dll Kmeans time = ', time.time() - color_reduce_time2)

    joined_img = np.hstack((original_without_dither, color_replaced_r, reduced_color_img))
    if not os.path.exists('A_STICHED_OUTPUT'):
        os.makedirs('A_STICHED_OUTPUT')
    cv2.imwrite('A_STICHED_OUTPUT/s' + str(segment_number) + '-' + str(sigma) + str(connectivity) + str(compactness) + 'c' + str(color_pockets) + name, joined_img)

    return output_img_path, reduced_color_path, str(time.time()-start), reduce_color_number


# filename, dither_flag, dither_color, segment_number, connectivity, compactness, sigma, color_pockets, resize_flag, resize_factor, reduce_color_number, dim_change_flag, dim
# a,b,c = main('E:/Work/dithered_color_percent/images/Miami-Heat.png',False, 8,100,False,3,2,4, True,3,6, True, (900,1200))

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
# print(a,b)
# path = dither('C:/Users/prasa/Downloads/Bishal/dithered_color_percent/images/Siam-Red-Pride.jpg',4)
# print(path)