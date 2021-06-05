# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/13/21 5:52 PM
# @Version	1.0
# --------------------------------------------------------
import os
import random
import shutil

training_dir = '/mnt/data/logo/training/images/train'
save_dir = './calib_imgs'
os.makedirs(save_dir, exist_ok=True)
images = os.listdir(training_dir)
images = [os.path.join(training_dir, img) for img in random.sample(images, 1000)]
for image in images:
    # print(image)
    shutil.copy(image, save_dir)


