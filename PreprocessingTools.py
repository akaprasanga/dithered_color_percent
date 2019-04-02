import cv2
import numpy as np
from PIL import Image
import os
from dither_from_dll import FunctionsFromDLL



class Preprocessing:

    def __init__(self):
        self.dllFunctions = FunctionsFromDLL()

        pass

    def resize_img_to_dim(self, img, dim):
        img = cv2.resize(img, dim)
        return img

    def dither(self, filename, number_of_color, dir_path):
        saving_path = self.dllFunctions.dither_image(filename, number_of_color, dir_path)
        name = filename.split('/')
        name = name[len(name) - 1]
        return saving_path + name

    def scale_up_image(self, img, factor):
        h, w = img.shape[:2]
        h = h * factor
        w = w * factor
        img = cv2.resize(img, (w, h))
        return img

    def main(self, filename, img_to_process, dither_flag, dither_color, segment_number, connectivity, compactness, sigma, color_pockets, resize_flag, resize_factor, reduce_color_number, dim_change_flag, dim, grayscale_flag):
        print('Processing started ==', filename)
        # print(img_to_process)
        dir_path = os.getcwd()
        dir_path = dir_path.replace('\\', '/')
        actual_name = os.path.splitext(filename)[0]
        actual_name = actual_name.split('/')
        actual_name = actual_name[len(actual_name) - 1]
        original_without_dither = cv2.imread(filename)

        if grayscale_flag == True:
            # im = cv2.imread(filename, 0)
            # if not os.path.exists('A_Grayscaled'):
            #     os.makedirs('A_Grayscaled')
            # path_after_grayscale = dir_path + '/A_Grayscaled/' + actual_name + '.png'
            # im = np.dstack((im, im, im))
            # cv2.imwrite(path_after_grayscale, im)
            # filename = path_after_grayscale
            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)

        no_dim_change = cv2.imread(filename)
        orig_dim = tuple((no_dim_change.shape[1], no_dim_change.shape[0]))
        if (dim_change_flag == True) and (dim != orig_dim):
            # dim_changed = cv2.resize(no_dim_change, dim)
            # if not os.path.exists('A_900X1200'):
            #     os.makedirs('A_900X1200')
            # path_after_dim_changed = dir_path + '/A_900X1200/' + actual_name + '.png'
            # cv2.imwrite(path_after_dim_changed, dim_changed)
            # filename = path_after_dim_changed
            img_to_process = cv2.resize(img_to_process, dim)
            original_without_dither = cv2.resize(original_without_dither, dim)


        if resize_flag == True:
            # non_resized = cv2.imread(filename)
            upscaled = self.scale_up_image(img_to_process, resize_factor)
            # upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
            # upscaled_pil = convert_to_pillow_format(upscaled_rgb)

            if not os.path.exists('A_UPSCALED_INPUT'):
                os.makedirs('A_UPSCALED_INPUT')
            path_after_upscaled = dir_path + '/A_UPSCALED_INPUT/' + actual_name + '.png'
            # upscaled_pil.save(path_after_upscaled)
            cv2.imwrite(path_after_upscaled, upscaled)
            filename = path_after_upscaled

        if resize_flag == False:
            if not os.path.exists('A_UPSCALED_INPUT'):
                os.makedirs('A_UPSCALED_INPUT')
            path_after_upscaled = dir_path + '/A_UPSCALED_INPUT/' + actual_name + '.png'
            # upscaled_pil.save(path_after_upscaled)
            cv2.imwrite(path_after_upscaled, img_to_process)
            filename = path_after_upscaled

        if dither_flag == True:
            p = filename
            try:
                path = self.dither(p, dither_color, dir_path)
                filename = path
            except ValueError:
                print('Dither failed due to less number of Colors in Image than defined color')
                dither_flag = False