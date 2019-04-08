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
from scipy.spatial import distance
from os import walk
import random
import pyvips as vip
import time
from sklearn.cluster import KMeans
from numba import types
from numba.typed import Dict



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

    def get_colors_from_img(self, filename, level):
        img = Image.open(filename).convert('HSV')
        c = img.getcolors()
        length = len(c)
        colors = [x[1] for x in c]
        colors_list=[]
        for i in range(0, 3):
            sorted_h = sorted(colors, key=lambda tup: tup[i])
            colors_list.append(sorted_h[level])
            colors_list.append(sorted_h[(length-1)-level])

        print(level, (length-1)-level)
        colors_list = list(set(colors_list))
        choosed_set = set(colors_list)
        remaining_colors = [x for x in colors if x not in choosed_set]

        return np.array(colors_list), np.array(remaining_colors)

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

    def match_nearest_colors(self, image, distance_threshold):
        img = Image.fromarray(image.astype('uint8'))
        c = img.getcolors()
        c = sorted(c, reverse=True)
        total_color = [x[1] for x in c]
        # print(c)
        # print(total_color)

        # creting distance table
        table = np.zeros((len(total_color), len(total_color)))
        for i, c1 in enumerate(total_color):
            for j, c2 in enumerate(total_color):
                if i >= j:
                    table[i, j] = 1000
                else:
                    table[i, j] = distance.euclidean(c1, c2)
        # print(table)

        replacing_candidate_dict = {}
        for i in range(table.shape[1]-1, 1, -1):
            minimum = np.min(table[:, i])
            if minimum < distance_threshold:
                index = np.where(table[:, i] == minimum)[0][0]
                replacing_candidate_dict[total_color[i]] = total_color[index]

        print(len(replacing_candidate_dict))
        return replacing_candidate_dict

    def remove_less_dominant_color(self, image, percentage_distn):
        image = Image.fromarray(image.astype('uint8'))
        img = image.copy()
        total_colors = img.getcolors()
        print(len(total_colors))
        w, h = img.size
        tolorance_distribution = w*h*percentage_distn/100
        sorted_colors = sorted(total_colors)
        low_tolerance_colors = [i[1] for i in sorted_colors if i[0] <= tolorance_distribution]
        all_color_list = [i[1] for i in total_colors]
        main_color = self.difference_of_list(all_color_list, low_tolerance_colors)

        if len(low_tolerance_colors)>0 and len(main_color):
            distance_2d_array = np.zeros((len(main_color), len(low_tolerance_colors)))
            for i, main in enumerate(main_color):
                for j, replace in enumerate(low_tolerance_colors):
                    distance_2d_array[i, j] = self.compute_ecludean_distance_of_tuple(main, replace)

            nearest_to_replacing = {}
            for i in range(0, distance_2d_array.shape[1]):
                index = np.where(distance_2d_array[:, i] == np.min(distance_2d_array[:, i]))
                nearest_to_replacing[low_tolerance_colors[i]] = main_color[index[0][0]]
            print('Shape ==',distance_2d_array.shape)
            print('Replacing==', len(nearest_to_replacing))

            black_img = image.copy()
            black_img_pixmap = black_img.load()
            for x in range(0, w):
                for y in range(0, h):
                    current_pixel = black_img_pixmap[x, y]
                    if current_pixel in nearest_to_replacing.keys():
                        black_img_pixmap[x, y] = nearest_to_replacing[current_pixel]
            print('Second from PIL', len(black_img.getcolors()))
            # black_img.save('replaced_minim.png')
            return np.asarray(black_img)

        else:
            print('operation not required')
            return np.asarray(image)

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
        picture = Image.open(filename).convert('HSV')
        width, height = picture.size
        picture_pixmap = picture.load()
        # print(replacing_dict[(11, 58, 221)])

        for x in range(0, width):
            for y in range(0, height):
                current_color = picture_pixmap[x, y]
                if current_color in replacing_dict:
                    picture_pixmap[x,y] = replacing_dict[current_color][random.randint(0, 2)]
                # try:
                #     picture_pixmap[x, y] = replacing_dict[current_color][random.randint(0, 2)]
                # except KeyError:
                #     pass
                # picture.putpixel((x, y), color_list[random.randint[]])

        print('Time =:', time.time()-start)
        picture = picture.convert('RGB')
        picture.save('trail_mix.png')
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

    def color_replacing_function(self, image, replacing_dict):
        img = Image.fromarray(image.astype('uint8'))
        print(replacing_dict)
        # logical refined dictionary
        refined = {}
        for k, v in replacing_dict.items():
            if v in replacing_dict:
                refined[k] = replacing_dict[v]
            else:
                refined[k] = v

        print(refined)
        replacing_dict = refined
        img_pixmap = img.load()
        w, h =img.size
        for x in range(0, w):
            for y in range(0, h):
                current_pixel = img_pixmap[x, y]
                if current_pixel in replacing_dict:
                    img_pixmap[x, y] = replacing_dict[current_pixel]

        img.save('reduced_near_colors.png')
        return np.asarray(img)

    def plot_hue_and_value(self, list_of_hue, list_of_value, list_of_rgb, list_of_saturation, labels):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i, val in enumerate(list_of_hue):
            hex_color = self.get_hex_from_rgb(list_of_rgb[i])
            ax.scatter(list_of_hue[i], list_of_value[i], list_of_saturation[i], c=hex_color, edgecolor=self.edge_color_dict[str(labels[i])] , linewidths= 2, s=80, label=labels[i])
            ax.grid(color='#AFEEEE', linestyle='--', linewidth=0.5)
        plt.xlabel('HUE')
        plt.ylabel('VALUE / Lightness')

    def create_new_axes_for_HSV(self, hue_list, sat_list, value_list):
        hmax, hmin = max(hue_list), min(hue_list)
        smax, smin = max(sat_list), min(sat_list)
        vmax, vmin = max(value_list), min(value_list)
        h_interval = (hmax-hmin)/len(hue_list)
        s_interval = (smax-smin)/len(sat_list)
        v_interval = (vmax-vmin)/len(value_list)

        print('hue max: ',hmax, 'Min : ',hmin, 'Interval :', h_interval)
        print('sat max: ', smax, 'Min : ', smin, 'Interval :', s_interval)
        print('value max: ', vmax, 'Min : ', vmin, 'Interval :', v_interval)

        # for i in range(1, len(hue_list)+1):
        #     print('Container ', i, '=', [hmin+h_interval*i, smin+s_interval*i, vmin+v_interval*i])
        # h_array = np.true_divide(np.array(hue_list)-hmin, h_interval)
        # s_array = np.true_divide(np.array(sat_list)-smin, s_interval)
        # v_array = np.true_divide(np.array(value_list)-vmin, v_interval)
        axis_interval = 64
        cube_num = 256/axis_interval

        h_array = np.true_divide(np.array(hue_list), axis_interval)
        s_array = np.true_divide(np.array(sat_list), axis_interval)
        v_array = np.true_divide(np.array(value_list), axis_interval)

        normalized_into_coordinates = np.dstack((h_array, s_array, v_array)).astype('uint8').reshape(-1, 3)

        named_number = np.zeros(normalized_into_coordinates.shape[0])
        for i in range(0, normalized_into_coordinates.shape[0]):
            # print(normalized_into_coordinates[i], normalized_into_coordinates[i, 1], normalized_into_coordinates[i,2])
            named_number[i] = normalized_into_coordinates[i, 0]+normalized_into_coordinates[i, 1]*cube_num+normalized_into_coordinates[i, 2]*cube_num*cube_num

        # cube_labels = np.mo
        # print(named_number)
        return list(h_array), list(s_array), list(v_array), named_number

    def create_combination_and_distance_table(self, main_colors, remaining_colors):
        main_colors_Y = np.multiply(main_colors, 0.67)
        main_colors_X = np.multiply(main_colors, 0.33)
        combination_table = np.zeros((main_colors.shape[0], main_colors.shape[0], 3))
        for i in range(0, main_colors.shape[0]):
            for j in range(0, main_colors.shape[0]):
                combination_table[i, j] = main_colors_Y[i]+main_colors_X[j]

        # replacing_dict = Dict.empty(
        #     key_type=types.float64[:],
        #     value_type=types.float64[:],
        # )
        replacing_list = []
        for i in range(0, remaining_colors.shape[0]):
            distance_table = np.zeros(combination_table.shape)
            distance_table[:, :, :] = remaining_colors[i]

            distance = np.linalg.norm(combination_table - distance_table, axis=2)
            index = np.where(distance == distance.min())
            y = index[0][0]
            x = index[1][0]
            # replacing_dict[tuple(remaining_colors[i])] = tuple((np.min(distance), main_colors[y], main_colors[y],main_colors[x]))
            replacing_list.append((remaining_colors[i],np.min(distance), main_colors[y], main_colors[y],main_colors[x]))
            # print(index[0][0], index[1][0])
        replacing_list = sorted(replacing_list, key=lambda tup: tup[1])
        return replacing_list

    def sort_and_pick(self, mix_ply_list, number_to_replace):
        required_list = mix_ply_list[:number_to_replace]
        # print(required_list)
        replacing_dict = {}
        for each in required_list:
            replacing_dict[tuple(each[0])] = tuple((tuple(each[2]), tuple(each[3]), tuple(each[4])))
        return replacing_dict

# if __name__ == '__main__':
#     mixPly = MixPLy()
#     filename = 'E:\Work\graph\s250-2False5c4Miami-Heat.processed.png'
#     img = Image.open(filename).convert('HSV')
#     i = Image.open(filename).convert('RGB')
#     rgb_colors = i.getcolors()
#     rgb_colors = [x[1] for x in rgb_colors]
#
#     c = img.getcolors()
#     total_color_number = len(c)
#     total_colors = [x[1] for x in c]
#
#     hue_list = [x[0] for x in total_colors]
#     sat_list = [x[1] for x in total_colors]
#     value_list = [x[2] for x in total_colors]
#
#     main_colors, remaining_colors = mixPly.get_colors_from_img(filename, 0)
#     mix_ply_list = mixPly.create_combination_and_distance_table(main_colors, remaining_colors)
#     replacing_dict = mixPly.sort_and_pick(mix_ply_list, number_to_replace=2)
#     mix_ply_img = mixPly.create_mixply_image(filename, replacing_dict)

