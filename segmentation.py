from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import numpy as np
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import os
import hitherdither
from PIL import Image
import cv2


class Segment:

    def __init__(self):
        pass

    def slic_superpixel(self, filename, image, segment_number, connectivity, s, k, color_pockets, resize_flag, resize_factor, dim_change_flag, dim, grayscale_flag):
        """
        :param filename: image to segment
        :param segment_number: nomuber of regions based on LAB color Space
        :return: segmented images with defined boundaries
        """

        print(image.shape)

        i = Image.fromarray(image)
        i.save('input.png')

        # if grayscale_flag == True:
        #     image = np.dstack((image, image, image))
        #     print(image.shape)

        image_r = image[:, :, :3]
        image_r = image_r.astype(float)
        segments_slic_r = slic(image_r, n_segments=segment_number, enforce_connectivity=connectivity, convert2lab=True,
                             multichannel=True, sigma=s, compactness=k)
        print(np.unique(segments_slic_r))

        channel_one = np.zeros(segments_slic_r.shape, segments_slic_r.dtype)
        channel_two = segments_slic_r // 256
        channel_three = segments_slic_r % 256

        three_channel = np.dstack((channel_one, channel_two, channel_three))

        three_channel = Image.fromarray(three_channel.astype('uint8'))
        # three_channel.save('segmented.png')
        fname = filename.split('/')
        fname = fname[len(fname)-1]
        if not os.path.exists('A_SEGMENTED'):
            os.makedirs('A_SEGMENTED')
        length = len(fname)
        fname = fname[0:length-4]+'-r'+fname[length-4:length]
        segmented_img_path = 'A_SEGMENTED/s' + str(segment_number) + '-' + str(s) + str(connectivity) + str(
            k) + 'c' + str(color_pockets) + fname
        # segments_slic_r_pil = Image.fromarray(segments_slic_r.astype('uint8'))
        three_channel.save(segmented_img_path)
        return segments_slic_r, segmented_img_path

    # def fz_superpixel(self, filename, f_scale):
    #     """
    #     :param filename: image to segment
    #     :param segment_number: nomuber of regions based on LAB color Space
    #     :return: segmented images with defined boundaries
    #     """
    #     image = img_as_float(io.imread(filename))
    #     image = image[:, :, :3]
    #     segments_fz = felzenszwalb(image, scale=f_scale, sigma=0.8)
    #
    #     boundaries = mark_boundaries(image, segments_fz)
    #     fname = filename.split('/')
    #     fname = fname[len(fname)-1]
    #     plt.imsave('segmented/fz' + str(f_scale) + fname, boundaries)
    #
    #     return segments_fz
    #
    #
    # def qc_superpixel(self, filename, r, m):
    #     """
    #     :param filename: image to segment
    #     :param segment_number: nomuber of regions based on LAB color Space
    #     :return: segmented images with defined boundaries
    #     """
    #     image = img_as_float(io.imread(filename))
    #     image = image[:, :, :3]
    #     segments_qc = quickshift(image, ratio=r, max_dist=m)
    #
    #     boundaries = mark_boundaries(image, segments_qc)
    #     fname = filename.split('/')
    #     fname = fname[len(fname)-1]
    #     plt.imsave('segmented/qc' + str(r) + fname, boundaries)
    #
    #     return segments_qc
