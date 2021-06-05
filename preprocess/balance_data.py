# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 6/3/21 10:50 AM
# @Version	1.0
# --------------------------------------------------------
import os
from tqdm import tqdm
import shutil
import json

os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.expanduser('~'), 'jobs')))
from cv_common.threads.mutli_process import MultiProcess


def all_files(video_path, result, path=True):
    if os.path.isfile(video_path):
        if not video_path.endswith('.json'):
            if path:
                result.append(video_path)
            else:
                result.append(os.path.basename(video_path))
        return
    for video_name in os.listdir(video_path):
        all_files(os.path.join(video_path, video_name), result, path)


def rm_data(imgs):
    rm_dir = '/mnt/data/logo/training'
    for image in tqdm(imgs):
        for path in os.popen('find {} -name "*{}*"'.format(rm_dir, '.'.join(image.split('.')[:-1]))).readlines():
            path = path.strip()
            command = 'rm -rf {}'.format(path)
            # print(command)
            os.system(command)


def find_path(imgs):
    label_dir = '/mnt/data/logo/training/labels/train'
    images_dir = '/mnt/data/logo/training/images/train'
    rm_root_dir = '/mnt/data/tmp/rm_dir'
    dir_num = 0
    rm_dir = os.path.join(rm_root_dir, str(dir_num))
    os.makedirs(rm_dir, exist_ok=True)
    for image in imgs:
        if len(os.listdir(rm_dir)) >= 1000:
            dir_num += 1
            rm_dir = os.path.join(rm_root_dir, str(dir_num))
            os.makedirs(rm_dir, exist_ok=True)

        name_base = '.'.join(image.split('.')[:-1])
        label_path = os.path.join(label_dir, '{}.txt'.format(name_base))
        img_path = os.path.join(images_dir, image)
        try:
            assert os.path.exists(img_path) and os.path.exists(label_path)
            shutil.move(img_path, rm_dir)
            shutil.move(label_path, rm_dir)
        except Exception as e:
            print("{}: {}: {}".format(label_path, img_path, e))

images = []
logo_generate_dir = '/mnt/data/TB/anno_make_watermark_tmp_lt'
all_files(logo_generate_dir, images, path=False)
# mp = MultiProcess(10)
# mp.set_job(find_path, imgs=images)
# mp.start()
