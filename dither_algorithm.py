import cv2
import numpy as np
from segmentation import Segment
import time
from PIL import Image
import os
from dither_from_dll import FunctionsFromDLL
from mixply import MixPLy

dllFunctions = FunctionsFromDLL()
seg = Segment()
mixPly = MixPLy()
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
#     ready_to_process = Image.open_filedialog(filename).convert('RGB')
#     color = ready_to_process.getcolors()
#     if color == None:
#         reduced_img, cluster_number = cluster.mini_batch_kmeans(filename, 24)
#         cv2.imwrite(filename,reduced_img)


def main(filename, img_to_process, dither_flag, dither_color, segment_number, connectivity, compactness, sigma, color_pockets, resize_flag, resize_factor, reduce_color_number, dim_change_flag, dim, grayscale_flag):
    print('Processing started ==', filename)
    # print(img_to_process)

    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\', '/')
    actual_name = os.path.splitext(filename)[0]
    actual_name = actual_name.split('/')
    actual_name = actual_name[len(actual_name) - 1]
    # original_without_dither = cv2.imread(filename)
    original_without_dither = np.asarray(Image.open(filename).convert('RGB'), dtype='uint8')

    if grayscale_flag == True:
        img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)

    no_dim_change = np.asarray(Image.open(filename).convert('RGB'), dtype='uint8')
    orig_dim = tuple((no_dim_change.shape[1], no_dim_change.shape[0]))
    if (dim_change_flag==True) and (dim != orig_dim):

        img_to_process = cv2.resize(img_to_process, dim)
        original_without_dither = cv2.resize(original_without_dither, dim)


    if resize_flag == True:
        upscaled = scale_up_image(img_to_process, resize_factor)

        if not os.path.exists('A_UPSCALED_INPUT'):
            os.makedirs('A_UPSCALED_INPUT')
        path_after_upscaled = dir_path + '/A_UPSCALED_INPUT/' + actual_name + '.png'

        upscaled = Image.fromarray(upscaled)
        upscaled.save(path_after_upscaled)
        filename = path_after_upscaled

    if resize_flag == False:
        if not os.path.exists('A_UPSCALED_INPUT'):
            os.makedirs('A_UPSCALED_INPUT')
        path_after_upscaled = dir_path + '/A_UPSCALED_INPUT/' + actual_name + '.png'

        img_to_process = Image.fromarray(img_to_process)
        img_to_process.save(path_after_upscaled)
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
    color_replaced_r = np.asarray(Image.open(color_replaced_r_path).convert('RGB'), dtype='uint8')
    if resize_flag == True:
        color_replaced_r = scale_down_image(color_replaced_r, resize_factor)
    name = filename.split('/')
    name = name[len(name)-1]
    if not os.path.exists('A_OUTPUT'):
        os.makedirs('A_OUTPUT')
    output_img_path = 'A_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name
    color_replaced_r = Image.fromarray(color_replaced_r)
    color_replaced_r.save(output_img_path)

    output_img_path = dir_path + '/' + output_img_path

    color_reduce_time2 = time.time()
    reduced_color_path = dllFunctions.reduce_color(output_img_path, reduce_color_number, dir_path)

    mixply_img_path  = 'A_MIXEDPLY_OUTPUT/s' + str(segment_number) + '-' + str(sigma) + str(connectivity) + str(compactness) + 'c' + str(color_pockets) + name
    if not os.path.exists('A_MIXEDPLY_OUTPUT'):
        os.makedirs('A_MIXEDPLY_OUTPUT')

    mapped_nearest_colors = mixPly.match_nearest_colors(np.asarray(Image.open(reduced_color_path).convert('RGB')), color_pockets)
    mapped_nearest_colors = Image.fromarray(mapped_nearest_colors)
    mapped_nearest_colors.save(mixply_img_path)
    mapped_nearest_colors.save(reduced_color_path)

    print('Dll Kmeans time = ', time.time() - color_reduce_time2)
    joined_img = np.hstack((original_without_dither, color_replaced_r, mapped_nearest_colors))
    if not os.path.exists('A_STICHED_OUTPUT'):
        os.makedirs('A_STICHED_OUTPUT')
    joined_img = Image.fromarray(joined_img)
    joined_img.save('A_STICHED_OUTPUT/s' + str(segment_number) + '-' + str(sigma) + str(connectivity) + str(compactness) + 'c' + str(color_pockets) + name)


    return output_img_path, reduced_color_path, str(time.time()-start), reduce_color_number, mixply_img_path


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