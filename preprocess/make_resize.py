import os
import random
import cv2
import numpy as np
from cv_common.threads.mutli_process import MultiProcess
from functools import partial


def all_files(video_path, result):
    if os.path.isfile(video_path):
        result.append(video_path)
        return
    for video_name in os.listdir(video_path):
        all_files(os.path.join(video_path, video_name), result)


def count_logo_data(logo_dir):
    logos = []
    all_files(logo_dir, logos)
    data_results = {}
    for logo_path in logos:
        logo_label = os.path.basename(logo_path).split('.')[0].split('_')[0]
        tmp = data_results.get(logo_label, [])
        tmp.append(logo_path)
        data_results[logo_label] = tmp
    return data_results


def ffmpeg_resize(input_img, save_path):
    command = 'ffmpeg -v quiet -y -i {} -s 640x640 {}'.format(input_img, save_path)
    os.system(command)


def generate_logo(imgs, save_dir):
    for img in imgs:
        ffmpeg_resize(img, os.path.join(save_dir, os.path.basename(img)))


def generate_logo_video_multi_threads():
    save_dir = '/mnt/ai_data/TB/background'
    img_dir = '/mnt/ai_data/suoshuai/Data/real_normal'
    imgs = []
    all_files(img_dir, imgs)
    os.makedirs(save_dir, exist_ok=True)
    mp = MultiProcess(30)
    func = partial(generate_logo, save_dir=save_dir)
    mp.set_job(func, imgs=imgs)
    mp.start()


def main():
    generate_logo_video_multi_threads()


if __name__ == '__main__':
    main()
