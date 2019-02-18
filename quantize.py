import numpy as np


def quantize_color(color_dict):
    num_colors = len(color_dict)
    quantized_color = {}
    key_list = []
    value_list = []
    for k,v in color_dict.items():
        key_list.append(k)
        value_list.append(v)

    np_values = np.array(value_list)
    quantize_bin = np.array([0, 0.25, 0.5, 0.75, 1])
    quantize_bin = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    digitized = np.digitize(np_values, quantize_bin)
    print(digitized)
    if (10 in digitized):
        print('100%')
    if ((9 in digitized) and (2 in digitized)):
        print("75% and 100%")
    elif((7 in digitized) or (8 in digitized)):

        if 3 in digitized:
            print('doen')
        else:
            ('what')


l1 = {(186, 175, 181): 0.012888992104040873, (135, 130, 156): 0.008476544356711566, (78, 78, 118): 0.023107292150487693, (28, 44, 103): 0.9555271713887599}
l2 = {(78, 78, 118): 0.001527133896918462, (28, 44, 103): 0.9984728661030815}
l3 = {(135, 130, 156): 0.00018724838498267952, (78, 78, 118): 0.02387416908529164, (28, 44, 103): 0.9759385825297257}
l4 = {(186, 175, 181): 0.7943908045977012, (135, 130, 156): 0.18997701149425286, (78, 78, 118): 0.015264367816091954, (28, 44, 103): 0.000367816091954023}
l5 = {(186, 175, 181): 0.09884778653729533, (135, 130, 156): 0.2889023650697392, (78, 78, 118): 0.46076409945421465, (28, 44, 103): 0.15148574893875075}
l6 = {(186, 175, 181): 0.2735160300497302, (135, 130, 156): 0.5671357528303883, (78, 78, 118): 0.12263252565866047, (28, 44, 103): 0.036715691461221035}


quantize_color(l1)
