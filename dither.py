from PIL import Image
import hitherdither

filename = 'IMG_20180612_0002_Page_2.png'
img = Image.open('images/IMG_20180612_0002_Page_2.png')
palette = hitherdither.palette.Palette.create_by_median_cut(img)
img_dithered = hitherdither.ordered.bayer.bayer_dithering(
    img, palette, [256/4, 256/4, 256/4], order=8)

img_dithered.save('dither'+filename)