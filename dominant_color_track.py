import cv2
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import os
from PIL import Image

def convert_to_pillow_format(img):
    return Image.fromarray(img.astype('uint8'), 'RGB')



def get_dominant_color(image_path, k=5):

    # print(image_path)
    image = cv2.imread(image_path)

    image_copy = image.copy()
    original = image.copy()
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    label_counts = Counter(labels)
    label_counts = label_counts.most_common()
    # print(label_counts)
    color_list = []
    cluster_number = []
    mean_color = {}
    for each in label_counts:
        cluster, area = each
        cluster_number.append(cluster)
        dominant_color = clt.cluster_centers_[cluster]
        mean_color[cluster] = clt.cluster_centers_[cluster]
        color_list.append(dominant_color)

    labels = np.reshape(labels, original.shape[:2])

    # checking and replacing with color
    # start = time.time()
    # for i in range(0, labels.shape[0]):
    #     for j in range(0, labels.shape[1]):
    #         center = labels[i][j]
    #         original[i][j] = list(mean_color[center])
    # print('tIME FOR DOUBLE FOR LOOP', time.time()-start)
    # orig = original.copy()
    for (i, segVal) in enumerate(np.unique(labels)):
        original[labels == segVal] = list(mean_color[segVal])  # Five start code for replacing values

    # original = cv2.resize(original, (0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    name = image_path.split('/')
    name = name[len(name)-1]
    if not os.path.exists('A_REDUCE_COLOR'):
        os.makedirs('A_REDUCE_COLOR')

    pil_name = 'A_REDUCE_COLOR/P' + name
    name = 'A_REDUCE_COLOR/' + name
    cv2.imwrite(name, original)
    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\','/')
    full_path = dir_path + '/' + name
    # print(len(mean_color))
    original_rgb = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
    pil_original = convert_to_pillow_format(original_rgb)
    pil_original = pil_original.convert('P', palette=Image.ADAPTIVE,colors = k)
    pil_original.save(pil_name)
    return full_path, len(mean_color), original


# image = cv2.imread('s100-1False3c3c-15Wizcraft 12 colors and 10 ratio.jpg')
# color = get_dominant_color('Wizcraft 12 colors and 10 ratio.png', k=5)
