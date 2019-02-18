from PIL import Image

img = Image.open('IMG_20180612_0002_Page_2.png')
print(img.getcolors())
img.save('trial.png')