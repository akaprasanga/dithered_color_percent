from PIL import Image
from os import walk


def main(filename):
    img = Image.open('New folder/'+filename).convert('RGB')
    img = img.convert('P', palette=Image.ADAPTIVE, dither = Image.FLOYDSTEINBERG,colors = 4)
    img.save('Out/'+filename)


f = []
for (dirpath, dirnames, filenames) in walk('New folder'):
    f.extend(filenames)
    break

counter = 0
for each in f:
    name = each
    main(name)
    print("processing image", counter)
    counter += 1