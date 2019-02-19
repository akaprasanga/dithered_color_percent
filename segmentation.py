from skimage.segmentation import slic,felzenszwalb,quickshift
from skimage.segmentation import mark_boundaries
import numpy as np
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

class Segment:

    def __init__(self):
        pass

    def slic_superpixel(self, filename, segment_number):
        """
        :param filename: image to segment
        :param segment_number: nomuber of regions based on LAB color Space
        :return: segmented images with defined boundaries
        """
        image = img_as_float(io.imread(filename))
        image = image[:, :, :3]
        segments_slic = slic(image, n_segments=segment_number, enforce_connectivity=False, convert2lab=True,
                             multichannel=True, sigma=3, compactness=3)

        boundaries = mark_boundaries(image, segments_slic)
        fname = filename.split('/')
        fname = fname[len(fname)-1]
        plt.imsave('segmented/slic' + str(segment_number) +fname, boundaries)

        return segments_slic, boundaries

    def fz_superpixel(self, filename, f_scale):
        """
        :param filename: image to segment
        :param segment_number: nomuber of regions based on LAB color Space
        :return: segmented images with defined boundaries
        """
        image = img_as_float(io.imread(filename))
        image = image[:, :, :3]
        segments_fz = felzenszwalb(image, scale=f_scale, sigma=0.8)

        boundaries = mark_boundaries(image, segments_fz)
        fname = filename.split('/')
        fname = fname[len(fname)-1]
        plt.imsave('segmented/fz' + str(f_scale) + fname, boundaries)

        return segments_fz


    def qc_superpixel(self, filename, r, m):
        """
        :param filename: image to segment
        :param segment_number: nomuber of regions based on LAB color Space
        :return: segmented images with defined boundaries
        """
        image = img_as_float(io.imread(filename))
        image = image[:, :, :3]
        segments_qc = quickshift(image, ratio=r, max_dist=m)

        boundaries = mark_boundaries(image, segments_qc)
        fname = filename.split('/')
        fname = fname[len(fname)-1]
        plt.imsave('segmented/qc' + str(r) + fname, boundaries)

        return segments_qc