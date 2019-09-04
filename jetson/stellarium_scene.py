import os

from PIL import Image

path = '/home/szymon/Pictures/Stellarium/test'
for f in os.listdir(path):
    img = Image.open(os.path.join(path, f))

    # remove bottom information box
    rect_size = (935-349, 900-876)
    rect_pos = (349, 876)
    rect = Image.new("RGBA", rect_size)
    img.paste(rect, rect_pos)

    # remove cross
    rect_size = (1, 468-432)
    rect_pos = (799, 432)
    rect = Image.new("RGBA", rect_size)
    img.paste(rect, rect_pos)

    rect_size = (818-782, 1)
    rect_pos = (782, 450)
    rect = Image.new("RGBA", rect_size)
    img.paste(rect, rect_pos)

    # # remove circle
    # rect_size = (818-782, 468-432)
    # rect_pos = (782, 432)
    # rect = Image.new("RGBA", rect_size)
    # img.paste(rect, rect_pos)

    # remove 10 degree FOV frame
    rect_size = (1, 900)
    rect_pos = (349, 0)
    rect = Image.new("RGBA", rect_size)
    img.paste(rect, rect_pos)

    rect_size = (900, 1)
    rect_pos = (349, 0)
    rect = Image.new("RGBA", rect_size)
    img.paste(rect, rect_pos)

    rect_size = (900, 1)
    rect_pos = (1249, 0)
    rect = Image.new("RGBA", rect_size)
    img.paste(rect, rect_pos)

    # crop to the 10 degree FOV (900x900 pixels)
    x1, x2, y1, y2 = 349, 1249, 0, 900  # cropping coordinates

    bg = Image.new('RGB', (x2 - x1, y2 - y1))
    bg.paste(img, (-x1, -y1))

    bg.save(os.path.join(
        path, "test_accuracy_{}.png".format(f.split(".")[0][-3:])))
