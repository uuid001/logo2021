# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 6/2/21 3:48 PM
# @Version	1.0
# --------------------------------------------------------
from PIL import Image
import os


def add_logo(logo_path):
    img = Image.open(logo_path)
    img = img.convert('RGBA')
    r, g, b, alpha = img.split()
    alpha = alpha.point(lambda i: i > 0 and 230)
    img.putalpha(alpha)
    img.save('test.png')


def test1(logo_path):
    img = Image.open(logo_path)
    img = img.convert("RGBA")
    width, height = img.size
    for i in range(0, width):
        for j in range(0, height):
            data = img.getpixel((i, j))
            # if sum(data[:3]) < sum([220, 220, 220]):
            #     img.putpixel((i, j), (255, 255, 255, 0))
            if sum(data[:3]) > sum([150, 150, 150]):
                img.putpixel((i, j), (255, 255, 255, 0))
            # else:
            #     print(data)
    img = img.resize((width, height), Image.ANTIALIAS)
    img.save(os.path.join('./refine', os.path.basename(logo_path)), quality=100)


def main():
    # logo_path = './ori/105_1.png'
    # logo_path = '/mnt/data/TB/TB_logo/refine/93_0.png'
    logo_path = '/mnt/data/TB/TB_logo/ori/forigen/93.png'
    test1(logo_path)


if __name__ == '__main__':
    main()
