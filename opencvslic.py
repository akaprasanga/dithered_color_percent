import cv2
import numpy as np

dir = 'E:/Work/images/Ascenure.png'
# image read
img = cv2.imread(dir) # BRG order, uint8 type

converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

num_superpixels = 4000  # desired number of superpixels
num_iterations = 100     # number of pixel level iterations. The higher, the better quality
prior = 5              # for shape smoothing term. must be [0, 5]
num_levels = 4
num_histogram_bins = 5 # number of histogram bins
height, width, channels = converted_img.shape
# seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
slic = cv2.ximgproc.createSuperpixelSLIC(img, 101, 50, 0)
slic._SuperpixelSplit()
# print(slic.dtype)
# mask  = slic.getLabelContourMask(True)
# cv2.imwrite('slic.jpg', mask)
#
slic.iterate(num_iterations)
num_of_superpixels_result = slic.getNumberOfSuperpixels()
print('Final number of superpixels: %d' % num_of_superpixels_result)
labels = slic.getLabels() # height
mask = slic.getLabelContourMask(False)
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
mask_inv = cv2.bitwise_not(mask)
result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
result = cv2.add(result_bg, result_fg)
cv2.imshow('ColorCodedWindow', result)
cv2.waitKey(0)


cv2.destroyAllWindows()
