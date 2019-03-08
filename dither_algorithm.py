import cv2
import numpy as np
from segmentation import Segment
import time
from PIL import Image
import quantize
import os
import dominant_color_track as cluster
import traceback

seg = Segment()
# area = 0
# area_r = 0

def resize_img_to_dim(img,dim):
    img = cv2.resize(img,dim)
    return img

def dither(filename, number_of_color, dim_change_flag, dim):
    img = cv2.imread(filename)
    # if dim_change_flag == True:
    #     img = resize_img_to_dim(img,dim)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img_dithered = img.convert('P', palette=Image.ADAPTIVE, dither=Image.FLOYDSTEINBERG,colors = number_of_color)


    # palette = hitherdither.palette.Palette.create_by_median_cut(img, n=number_of_color)
    # img_dithered = hitherdither.ordered.bayer.bayer_dithering(
    #     img, palette, 10, order=2)
    # print("dithered Color =", len(img_dithered.getcolors()))
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
    # if colors == None:
    #     image = image.convert('P', palette=Image.ADAPTIVE, colors=16)
    #     colors = image.getcolors()
        # cv2.imwrite('Error/'+str(random.randint(1,10000))+'.jpg', img)

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
        # max_value = get_max_value_from_dict(color_dist)
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
    # final2 = np.dstack((mask_b, mask_g, mask_r))

    return final1


def main(filename, dither_flag, dither_color, segment_number, connectivity, compactness, sigma, color_pockets, resize_flag, resize_factor, reduce_color_number, dim_change_flag, dim):
    print('Processing started ==', filename)
    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\', '/')
    actual_name = os.path.splitext(filename)[0]
    actual_name = actual_name.split('/')
    actual_name = actual_name[len(actual_name) - 1]

    no_dim_change = cv2.imread(filename)
    orig_dim = tuple((no_dim_change.shape[1], no_dim_change.shape[0]))

    if (dim_change_flag == True) and (dim != orig_dim):
        dim_changed = cv2.resize(no_dim_change, dim)
        if not os.path.exists('A_900X1200'):
            os.makedirs('A_900X1200')
        path_after_dim_changed = dir_path + '/' + 'A_900X1200/' + actual_name + '.png'
        cv2.imwrite(path_after_dim_changed, dim_changed)
        filename = path_after_dim_changed

    original_without_dither = cv2.imread(filename)

    if resize_flag == True:
        non_resized = cv2.imread(filename)
        upscaled = scale_up_image(non_resized,resize_factor)
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        upscaled_pil = convert_to_pillow_format(upscaled_rgb)
        c = upscaled_pil.getcolors()
        # print(c)
        if c == None:
            # print('came to reduce colors')
            upscaled_pil = upscaled_pil.convert('P', palette = Image.ADAPTIVE, colors=20)

        if not os.path.exists('A_UPSCALED_INPUT'):
            os.makedirs('A_UPSCALED_INPUT')
        path_after_upscaled = dir_path + '/' + 'A_UPSCALED_INPUT/' + actual_name + '.png'
        upscaled_pil.save(path_after_upscaled)
        # cv2.imwrite(path_after_upscaled, upscaled)
        filename = path_after_upscaled

    # if dim_change_flag == True:
    #     original_without_dither = resize_img_to_dim(original_without_dither, dim)
    start = time.time()
    if dither_flag == True:
        p = filename
        f_name, file_extension = os.path.splitext(p)
        if file_extension == '.jpg' or file_extension == '.jpeg' or file_extension == '.JPG':
            temp = cv2.imread(p)
            p = f_name + '.png'
            cv2.imwrite(p, temp)
        try:
            path = dither(p, dither_color, dim_change_flag, dim)
            filename = path
        except:
            traceback.print_exc()
            print('dither failed')
            filename = path_after_upscaled
            dither_flag = False
    segmented_r, segmented_img_path = seg.slic_superpixel(filename, segment_number,connectivity,sigma,compactness, color_pockets, resize_flag, resize_factor,dim_change_flag,dim)

    # print('fname before segval =', filename)
    original = cv2.imread(filename)

    original_r = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    color_replaced_r = count_proportion(segmented_r, original_r, color_pockets)

    if resize_flag == True:
        color_replaced_r = scale_down_image(color_replaced_r, resize_factor)
    name = filename.split('/')
    name = name[len(name)-1]
    if not os.path.exists('A_OUTPUT'):
        os.makedirs('A_OUTPUT')
    output_img_path = 'A_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name
    cv2.imwrite(output_img_path, color_replaced_r)
    # segmented_img_path = 'A_SEGMENTED/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets) + name
    segmented_image = cv2.imread(segmented_img_path)
    dithered_img_path = 'A_DITHERED/' + name
    dithered_img = cv2.imread(dithered_img_path)
    if resize_flag == True:
        # color_replaced_r = scale_down_image(color_replaced_r, resize_factor)
        segmented_image = scale_down_image(segmented_image, resize_factor)
        if dither_flag == True:
            dithered_img = scale_down_image(dithered_img, resize_factor)


    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\','/')
    output_img_path = dir_path + '/' + output_img_path
    dithered_img_path = dir_path + '/' + dithered_img_path
    segmented_img_path = dir_path + '/' + segmented_img_path
    reduced_color_path, cluster_number,reduced_color_img = cluster.get_dominant_color(output_img_path,reduce_color_number)

    try:
        # print('Shapes ==',original_without_dither.shape,segmented_image.shape, dithered_img.shape, color_replaced_r.shape)
        joined_img = np.hstack((original_without_dither,color_replaced_r,reduced_color_img))
        if not os.path.exists('A_STICHED_OUTPUT'):
            os.makedirs('A_STICHED_OUTPUT')
        cv2.imwrite('A_STICHED_OUTPUT/s' + str(segment_number) + '-' + str(sigma) + str(connectivity) + str(compactness) + 'c' + str(color_pockets) + name, joined_img)
    except:
        joined_img = np.hstack((original_without_dither,color_replaced_r,reduced_color_img))
        if not os.path.exists('A_STICHED_OUTPUT'):
            os.makedirs('A_STICHED_OUTPUT')
        cv2.imwrite('A_STICHED_OUTPUT/s' + str(segment_number)+'-'+ str(sigma)+str(connectivity)+str(compactness)+'c'+str(color_pockets)+name, joined_img)
    # print("Time Elapsed = ", time.time()-start)

    if dither_flag == True:
        return output_img_path, reduced_color_path, str(time.time()-start),cluster_number
    else:
        return output_img_path, reduced_color_path, str(time.time()-start),cluster_number


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