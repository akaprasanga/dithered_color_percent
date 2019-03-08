from PIL import Image
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io
import traceback

def convert_to_pillow_format(img):
    return Image.fromarray(img.astype('uint8'), 'RGB')


img = cv2.imread('Ascenure.processed.png')
# print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_before = convert_to_pillow_format(img)
print('Before ::', img_before.getcolors())
image_rescaled = rescale(img, 3, anti_aliasing=False)
# image_rescaled = convert_to_pillow_format(image_rescaled)
# print('After == ', image_rescaled.getcolors())
io.imsave('rescaled.jpg', image_rescaled)
# cv2.imwrite('rescaled.jpg', image_rescaled)
rescaled = Image.open('rescaled.jpg')
converted = img.convert('P', palette=Image.ADAPTIVE, colors = 16)
converted.save('converted.png')
print(len(converted.getcolors()))
traceback.print_exc()