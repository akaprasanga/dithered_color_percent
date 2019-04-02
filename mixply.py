from PIL import Image
import numpy as np
import pandas as pd
import webcolors
import matplotlib.pyplot as plt
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from mpl_toolkits.mplot3d import axes3d
import cv2
from os import walk
import random
import pyvips as vip
import time


class MixPLy:


    def __init__(self):
        self.edge_color_dict = {'0': 'red',
                  '1': 'blue',
                  '2': 'green',
                  '3': 'black',
                  '4': 'cyan',
                  '5': 'grey',
                  '6': 'aqua',
                    '7': 'brown',
                    '8': 'marron',
                    '9': 'yellow'}


    def remove_area_from_list(self, list_with_area):
        list_without_area = []
        for each in list_with_area:
            area, color_tuple = each
            list_without_area.append(color_tuple)
        return list_without_area

    def get_hex_from_rgb(self, triplet):
        # print(triplet)
        return webcolors.rgb_to_hex(triplet)

    def get_colors_from_img(self, img, number_of_main_colors):
        c = img.getcolors()
        c = sorted(c)
        main_colors = c[-number_of_main_colors:]
        remaining_color = list(set(c) - set(main_colors))
        return self.remove_area_from_list(main_colors), self.remove_area_from_list(remaining_color)

    def multiply_all(self, l, factor):
        multiplied_list = []
        for each in l:
            each = tuple(ti * factor for ti in each)
            multiplied_list.append(each)
        return multiplied_list

    def create_list_of_tuple(self, array_list):
        number_of_elements = array_list.shape[0]
        list_of_tuple = []
        for i in range(0, number_of_elements, 3):
            list_of_tuple.append((array_list[i], array_list[i+1], array_list[i+2]))
        return list_of_tuple

    def create_mix_ply_table(self, main_color_list, number_of_ply=3):
        multiplier_small = 1 / number_of_ply
        multiplier_large = 1 - multiplier_small

        column = ['Color-A', 'Color-A%', 'Color-B', 'Color-B%', 'Mixed-Color']
        mix_ply_table = pd.DataFrame(columns=column)

        for i, c in enumerate(main_color_list):
            temp_list = main_color_list.copy()
            a = temp_list[i]
            del temp_list[i]
            a = [a]*len(temp_list)
            color_a_list = a.copy()
            color_b_list = temp_list.copy()
            a = np.array(list(sum(a, ())))
            temp_list = np.array(list(sum(temp_list, ())))
            multiplied_a = a * multiplier_small
            multiplied_temp_list = temp_list * multiplier_large
            mixed_list = multiplied_a + multiplied_temp_list
            mixed_list = self.create_list_of_tuple(mixed_list)

            for j, each in enumerate(mixed_list):
                new_row = {'Color-A': color_a_list[j], 'Color-A%': multiplier_small, 'Color-B': color_b_list[j], 'Color-B%': multiplier_large, 'Mixed-Color': each}
                mix_ply_table = mix_ply_table.append([new_row], ignore_index=True)

        return mix_ply_table

    def compute_ecludean_distance_of_tuple(self, color_one, color_two):

        color_one = tuple(ti / 255 for ti in color_one)
        color_two = tuple(ti / 255 for ti in color_two)
        color_one = sRGBColor(color_one[0], color_one[1], color_one[2])
        color_two = sRGBColor(color_two[0], color_two[1], color_two[2])

        color1_lab = convert_color(color_one, LabColor)

        color2_lab = convert_color(color_two, LabColor)

        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e

    def create_distance_table(self, mixed_ply_table, remaining_color_list):
        distance_table = pd.DataFrame(columns=['Color-A', 'Color-A%', 'Color-B', 'Color-B%', 'Mixed-Color', 'Color-R', 'Distance'])
        for each in remaining_color_list:
            for index, row in mixed_ply_table.iterrows():
                pd_rows = list(row.values)
                mixed_color = pd_rows[-1]
                distance = self.compute_ecludean_distance_of_tuple(mixed_color, each)
                new_row = {'Color-A': pd_rows[0], 'Color-A%': pd_rows[1], 'Color-B': pd_rows[2], 'Color-B%': pd_rows[3], 'Mixed-Color': pd_rows[4], 'Color-R':each, 'Distance':distance}
                distance_table = distance_table.append([new_row], ignore_index=True)
        return  distance_table

    def visualize_combination(self, distance_table, number_of_values, filename):
        list_of_pallates = []
        sorted_df = distance_table.sort_values(by=['Distance']).head(number_of_values)
        sorted_df = sorted_df.reset_index(drop=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(sorted_df)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        for index, row in sorted_df.iterrows():
            img_1 = np.zeros((100, 100, 3), dtype='uint8')
            img_2 = np.zeros((100, 100, 3), dtype='uint8')
            img_3 = np.zeros((100, 100, 3), dtype='uint8')
            img_4 = np.zeros((100, 100, 3), dtype='uint8')


            row = list(row.values)
            main_color_1 = row[0]
            main_color_2 = row[2]
            main_color_3 = row[5] # target color
            main_color_4 = row[4] # created color using mixply
            distance = row[6]
            main_color_4 = tuple(int(round(ti)) for ti in main_color_4)


            img_1[:, :, 0] = main_color_1[0]
            img_1[:, :, 1] = main_color_1[1]
            img_1[:, :, 2] = main_color_1[2]

            img_2[:, :, 0] = main_color_2[0]
            img_2[:, :, 1] = main_color_2[1]
            img_2[:, :, 2] = main_color_2[2]

            img_3[:, :, 0] = main_color_3[0]
            img_3[:, :, 1] = main_color_3[1]
            img_3[:, :, 2] = main_color_3[2]

            img_4[:, :, 0] = main_color_4[0]
            img_4[:, :, 1] = main_color_4[1]
            img_4[:, :, 2] = main_color_4[2]
            final = np.hstack((img_1, img_2, img_3, img_4))

            font = cv2.FONT_HERSHEY_COMPLEX_SMALL

            cv2.putText(final, str(distance)[:6], (0, (final.shape[0]//2)-20), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            final = cv2.line(final,(0, final.shape[0]-2), (final.shape[1], final.shape[0]), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            list_of_pallates.append(final)
            # plt.imsave(str(distance)+'.jpg', final)

        # for i, each in enumerate(list_of_pallates):
        f = np.vstack((list_of_pallates[0:]))
        plt.imsave(str(1) + '.jpg', f)

    def create_index_file(self, main_colors, remaining_colors, total_colors_num, distance_table, number_of_mixply_color):
        sorted_df = distance_table.sort_values(by=['Distance']).head(number_of_mixply_color)
        sorted_df = sorted_df.reset_index(drop=True)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(sorted_df)

        mixply_dict = {}
        mean_dict = {}

        for index, row in sorted_df.iterrows():
            key, color1, color2, mixed_color = row['Color-R'], row['Color-A'], row['Color-B'], row['Mixed-Color']
            if key not in mixply_dict:
                mixply_dict[key] = [color1, color2, color2]
            if key not in mean_dict:
                mean_dict[key] = mixed_color

        # for solid colors
        # print('Mix ply Dict:', len(mixply_dict))
        temp_list = main_colors.copy() + remaining_colors.copy()
        for each in temp_list:
            if each not in mixply_dict:
                mixply_dict[each] = [each, each, each]
            if each not in mean_dict:
                mean_dict[each] = each
        # print('Length final:', len(mixply_dict))
        # print(mixply_dict)
        return mixply_dict, mean_dict

    def difference_of_list(self, list1, list2):
        return list(set(list1) - set(list2))

    def create_mapping_table(self, distance_table, main_color_list, remaining_color, distance_threshold):
        sorted_df = distance_table.sort_values(by=['Distance'])
        sorted_df = sorted_df.reset_index(drop=True)
        # print(sorted_df)

        replacing_color_dict = {}

        # mapping solid colors
        for i in remaining_color:
            for j in main_color_list:
                d = self.compute_ecludean_distance_of_tuple(i, j)
                if d <= distance_threshold:
                    replacing_color_dict[i] = [j, j, j]

        # mapping mixply colors
        list_of_solid_colors = list(replacing_color_dict.keys())
        remaining_color = self.difference_of_list(remaining_color, list_of_solid_colors)

        remaining_colors


    def replace_with_mixed_color(self, filename, mean_color_dict):
        img = vip.Image.new_from_file(filename)
        img = img.flatten()
        for k, v in mean_color_dict.items():
            img = (img == k).ifthenelse(v, img)
        img.write_to_file('vips.png')

    def create_mixply_image(self, filename, replacing_dict):
        # img = cv2.imread(filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # color_list = []
        import time
        start = time.time()
        picture = Image.open(filename).convert('RGB')
        width, height = picture.size

        for x in range(0, width):
            for y in range(0, height):
                current_color = picture.getpixel((x, y))
                picture.putpixel((x,y), replacing_dict[current_color][random.randint(0, 2)])
                # picture.putpixel((x, y), color_list[random.randint[]])
        print('Time =:', time.time()-start)
        # picture.save('trail_mix.png')
        return np.asarray(picture)

    def call_mixply_function(self, number_of_colors, filename):
        start = time.time()
        img = Image.open(filename).convert('RGB')
        main_colors, remaining_colors = self.get_colors_from_img(img, number_of_main_colors=number_of_colors)
        mixed_ply_table = self.create_mix_ply_table(main_colors, 3)
        distance_table = self.create_distance_table(mixed_ply_table, remaining_colors)
        replacing_dict, mean_dict = self.create_index_file(main_colors, remaining_colors,
                                                             len(main_colors) + len(remaining_colors), distance_table,
                                                             30)
        mixed_ply_img = self.create_mixply_image(filename, replacing_dict)
        print('Mixply Time:', time.time()-start)
        return mixed_ply_img

if __name__ == '__main__':
    mixPly = MixPLy()
    filename = 'E:\Work\graph\s250-2False5c4Miami-Heat.processed.png'
    img = Image.open(filename).convert('RGB')
    main_colors, remaining_colors = mixPly.get_colors_from_img(img, number_of_main_colors=6)
    mixed_ply_table = mixPly.create_mix_ply_table(main_colors, 3)
    distance_table = mixPly.create_distance_table(mixed_ply_table, remaining_colors)
    mixPly.create_mapping_table(distance_table, main_colors, remaining_colors, 0.2)
    # replacing_dict, mean_dict = mixPly.create_index_file(main_colors, remaining_colors, len(main_colors)+len(remaining_colors), distance_table, 15)
    # mixPly.create_mixply_image(filename, replacing_dict)
    # mixPly.replace_with_mixed_color(filename, mean_dict)


    #
    #
    # f = []
    # for (dirpath, dirnames, filenames) in walk('images'):
    #     f.extend(filenames)
    #     break
    #
    # counter = 0
    # for each in f:
    #     # name = each
    #     print("processing image", counter)
    #     counter += 1
    #
    #     filename = each
    #
