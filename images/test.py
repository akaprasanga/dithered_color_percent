from PIL import Image

img = Image.open('Wizcraft 12 colors and 10 ratio.processed.png')

upscaled_pil = img.convert('P', palette=Image.ADAPTIVE, colors=20)
upscaled_pil.save('test.png')