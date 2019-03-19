import numpy as np
import cv2
from collections import Counter
from PIL import Image

img = cv2.imread('Xepe Gexote.png')
img = img[:, :, :3]
index = cv2.imread('temp.png', 0)
# colors_index = Counter([tuple(colors) for i in index for colors in i])
# print(colors_index)

# image = Image.fromarray(index.astype('uint8'), 'RGB')
# colors = image.getcolors()
# print(colors)

for j,val in enumerate(np.unique(index)):
    li = np.argwhere(index==val)
    # print(li)
    print(np.average(img[li]))
    # zeros[index==val] = 1
    # print(zeros.shape, img.shape)
    # segment = np.bitwise_and(zeros, img)
    # colors = Counter([tuple(colors) for i in segm ent for colors in i])
    # print(colors)

# colors = Counter([tuple(colors) for i in img for colors in i])
# print(colors)