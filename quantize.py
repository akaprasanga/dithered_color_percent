import numpy as np
import operator
from collections import OrderedDict
from operator import itemgetter
from collections import Counter


def quantize_into_pockets(color_dict, pocket_number=3):
    keys_list = []
    values_list = []
    bucket = []
    for k,v in color_dict.items():
        keys_list.append(k)
        values_list.append(v)
    pockets_size = 1 / pocket_number
    # color_dict = {(186, 175, 181): 0.012888992104040873, (135, 130, 156): 0.008476544356711566, (78, 78, 118): 0.023107292150487693, (28, 44, 103): 0.9555271713887599}
    # print(sorted_x)
    for i in range (0, pocket_number):
        sorted_x = sorted(color_dict.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_x)
        bucket.append(sorted_x[0][0])
        empty_list = {}
        for i, each in enumerate(sorted_x):
            if i == 0:
                color, distribution = each
                distribution = distribution - pockets_size
                # each = tuple((color,distribution))
                empty_list[color] = distribution
            else:
                color, distribution = each
                empty_list[color] = distribution
        color_dict = empty_list
    z = Counter(bucket)
    final_dict = {}
    for k,v in z.items():
        v = v * pockets_size
        final_dict[k] = v
    # print(final_dict)
    return final_dict


def rgb_to_cmy(color_dict):
    cmy_dict = {}
    for k, v in color_dict.items():
        cmy = tuple(ti / 255 for ti in k)
        cmy = tuple(1-ti for ti in cmy)
        cmy_dict[cmy] = v

    return cmy_dict


def cmy_to_rgb(val):
    val = 1 - val
    val = int(val * 255)
    return val


l1 = {(186, 175, 181): 0.012888992104040873, (135, 130, 156): 0.008476544356711566, (78, 78, 118): 0.023107292150487693, (28, 44, 103): 0.9555271713887599}
l2 = {(78, 78, 118): 0.001527133896918462, (28, 44, 103): 0.9984728661030815}
l3 = {(135, 130, 156): 0.00018724838498267952, (78, 78, 118): 0.02387416908529164, (28, 44, 103): 0.9759385825297257}
l4 = {(186, 175, 181): 0.7943908045977012, (135, 130, 156): 0.18997701149425286, (78, 78, 118): 0.015264367816091954, (28, 44, 103): 0.000367816091954023}
l5 = {(186, 175, 181): 0.09884778653729533, (135, 130, 156): 0.2889023650697392, (78, 78, 118): 0.46076409945421465, (28, 44, 103): 0.15148574893875075}
l6 = {(186, 175, 181): 0.2735160300497302, (135, 130, 156): 0.5671357528303883, (78, 78, 118): 0.12263252565866047, (28, 44, 103): 0.036715691461221035}

#
# quantize_into_pockets(l6)
