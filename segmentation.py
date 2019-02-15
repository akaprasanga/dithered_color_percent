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
                             multichannel=True, sigma=5, compactness=5)

        boundaries = mark_boundaries(image, segments_slic)
        fname = filename.split('/')
        fname = fname[len(fname)-1]
        plt.imsave('bound' + fname, boundaries)

        return segments_slic