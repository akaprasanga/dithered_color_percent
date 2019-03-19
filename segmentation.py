from skimage.segmentation import slic,felzenszwalb,quickshift
from skimage.segmentation import mark_boundaries
from skimage.transform import rescale
from skimage.transform import resize
import numpy as np
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import os
import hitherdither
from PIL import Image
import time
from skimage.measure import regionprops
from multiprocessing import Pool, TimeoutError


class Segment:

    def __init__(self):
        pass

    def slic_superpixel(self, filename, segment_number, connectivity, s, k, color_pockets, resize_flag, resize_factor, dim_change_flag, dim, grayscale_flag):
        """
        :param filename: image to segment
        :param segment_number: nomuber of regions based on LAB color Space
        :return: segmented images with defined boundaries
        """
        # print('Before Segmentation == ', filename)
        image = img_as_float(io.imread(filename))
        # if dim_change_flag == True:
        #     w, h = dim
        #     image =resize(image,(h,w),anti_aliasing=True)

        if grayscale_flag == True:
            image = np.dstack((image, image, image))

        image_r = image[:, :, :3]

        segmentation_time = time.time()

        segments_slic_r = slic(image_r, n_segments=segment_number, enforce_connectivity=connectivity, convert2lab=True,
                             multichannel=True, sigma=s, compactness=k)

        # with Pool(processes=4) as pool:
        #     pool.map(slic,[])
        end_seg_time = time.time()
        # temp_segmented = segments_slic_r * 10
        # io.imsave('temp.png', segments_slic_r)
        print('Segmentation Time = ', end_seg_time-segmentation_time)

        # boundaries = mark_boundaries(image, segments_slic)
        # boundaries_r = mark_boundaries(image_r, segments_slic_r)

        fname = filename.split('/')
        fname = fname[len(fname)-1]
        if not os.path.exists('A_SEGMENTED'):
            os.makedirs('A_SEGMENTED')
        length = len(fname)
        fname = fname[0:length-4]+'-r'+fname[length-4:length]
        segmented_img_path = 'A_SEGMENTED/s' + str(segment_number) + '-' + str(s) + str(connectivity) + str(
            k) + 'c' + str(color_pockets) + fname
        io.imsave(segmented_img_path, segments_slic_r)


        # out = image_r.copy()
        #
        # def paint_region_with_avg_intensity(rp, mi, channel):
        #     for i in range(rp.shape[0]):
        #         out[rp[i][0]][rp[i][1]][channel] = mi
        # for i in range(3):
        #     regions = regionprops(segments_slic_r, intensity_image=out[:, :, i])
        #     for r in regions:
        #         paint_region_with_avg_intensity(r.coords, int(r.mean_intensity), i)
        # io.imsave('trial.jpg', out)

        return segments_slic_r, segmented_img_path

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

    def dither(filename):
        img = Image.open(filename)
        palette = hitherdither.palette.Palette.create_by_median_cut(img, n=4)
        # palette = hitherdither.palette.Palette(
        #     [0x080000, 0x201A0B, 0x432817, 0x492910,
        #      0x234309, 0x5D4F1E, 0x9C6B20, 0xA9220F,
        #      0x2B347C, 0x2B7409, 0xD0CA40, 0xE8A077,
        #      0x6A94AB, 0xD5C4B3, 0xFCE76E, 0xFCFAE2])
        img_dithered = hitherdither.ordered.bayer.bayer_dithering(
            img, palette, 10, order=8)
        name = filename.split['/']
        name = name[len(name)-1]
        dithered_img_path = filename.split('/')
        dithered_img_path = dithered_img_path.pop()
        path_string = ''
        for each in dithered_img_path:
            path_string = each+path_string
        path_string = path_string + 'A_DITHERED/' + name
        if not os.path.exists('A_DITHERED'):
            os.makedirs('A_DITHERED')
        img_dithered.save(path_string)
        return path_string
