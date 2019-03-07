from PIL import Image
img = Image.open('001109_0552_0122_nlhs.jpg')
# print(img)
color = img.getcolors(img.size[0]*img.size[1])
print(len(color))