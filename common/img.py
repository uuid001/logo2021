# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 6/2/21 6:15 PM
# @Version	1.0
# --------------------------------------------------------
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def random_brightness(img, ratio):
    """

    :param img:
    :param ratio:
    :return:
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bright_enhancer = ImageEnhance.Brightness(img)
    img = bright_enhancer.enhance(ratio)
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def random_contrast(img, ratio):
    """

    :param img:
    :param ratio:
    :return:
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(ratio)
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img
