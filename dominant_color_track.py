import cv2
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
import os
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import time


def convert_to_pillow_format(img):
    return Image.fromarray(img.astype('uint8'), 'RGB')


def mini_batch_kmeans(image_path, k):
    filename = image_path
    # k = 10
    image = cv2.imread(filename)
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    final_colors = len(Counter(labels))
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    # image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)


    # display the images and wait for a keypress
    # cv2.imshow("image", np.hstack([image, quant]))
    # cv2.waitKey(0)
    return quant, final_colors


def get_dominant_color(image_path, k=5):

    # print(image_path)
    # start = time.time()
    # image = cv2.imread(image_path)
    #
    # image_copy = image.copy()
    # original = image.copy()
    # # reshape the image to be a list of pixels
    # image = image.reshape((image.shape[0] * image.shape[1], 3))
    #
    # # cluster and assign labels to the pixels
    # clt = KMeans(n_clusters=k)
    # labels = clt.fit_predict(image)
    #
    # label_counts = Counter(labels)
    # label_counts = label_counts.most_common()
    # # print(label_counts)
    # color_list = []
    # cluster_number = []
    # mean_color = {}
    # for each in label_counts:
    #     cluster, area = each
    #     cluster_number.append(cluster)
    #     dominant_color = clt.cluster_centers_[cluster]
    #     mean_color[cluster] = clt.cluster_centers_[cluster]
    #     color_list.append(dominant_color)
    #
    # labels = np.reshape(labels, original.shape[:2])
    #
    # for (i, segVal) in enumerate(np.unique(labels)):
    #     original[labels == segVal] = list(mean_color[segVal])  # Five start code for replacing values

    k_means_predicted, final_colors = mini_batch_kmeans(image_path, k)
    name = image_path.split('/')
    name = name[len(name)-1]
    if not os.path.exists('A_REDUCE_COLOR'):
        os.makedirs('A_REDUCE_COLOR')

    # pil_name = 'A_REDUCE_COLOR/P' + name
    name = 'A_REDUCE_COLOR/' + name
    cv2.imwrite(name, k_means_predicted)
    dir_path = os.getcwd()
    dir_path = dir_path.replace('\\', '/')
    full_path = dir_path + '/' + name
    # print(len(mean_color))
    # original_rgb = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)
    # pil_original = convert_to_pillow_format(original_rgb)
    # pil_original = pil_original.convert('P', palette=Image.ADAPTIVE,colors = k)
    # pil_original.save(pil_name)
    return full_path, final_colors, k_means_predicted

# image = cv2.imread('s100-1False3c3c-15Wizcraft 12 colors and 10 ratio.jpg')
# filename = 'images/Ascenure.processed.png'
# a = 10
#
# original = cv2.imread(filename)
# path , t, img = get_dominant_color(filename, a)
