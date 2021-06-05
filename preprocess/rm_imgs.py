# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/30/21 3:19 PM
# @Version	1.0
# --------------------------------------------------------
import os
from tqdm import tqdm

os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.expanduser('~'), 'jobs')))
from cv_common.threads.mutli_process import MultiProcess


def all_files(video_path, result):
    if os.path.isfile(video_path):
        result.append(os.path.basename(video_path))
        return
    for video_name in os.listdir(video_path):
        all_files(os.path.join(video_path, video_name), result)


def rm_data(imgs):
    rm_dir = '/mnt/data/logo/training'
    for image in tqdm(imgs):
        for path in os.popen('find {} -name "*{}*"'.format(rm_dir, '.'.join(image.split('.')[:-1]))).readlines():
            path = path.strip()
            command = 'rm -rf {}'.format(path)
            # print(command)
            os.system(command)


images = []
logo_generate_dir = '/mnt/data/TB/anno_make_watermark_tmp'
all_files(logo_generate_dir, images)

mp = MultiProcess()
mp.set_job(rm_data, imgs=images)
mp.start()
