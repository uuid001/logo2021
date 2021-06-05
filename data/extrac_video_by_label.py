# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/31/21 3:40 PM
# @Version	1.0
# --------------------------------------------------------
import os
import shutil
from tqdm import tqdm


def read_label_by_txt(txt_path):
    """

    :param txt_path:
    :return:
    """
    results = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('|')
            video_name = line[2]
            tmp = results.get(video_name, [])
            tmp.append((line[3], line[4], line[6], line[5]))
            results[video_name] = tmp
    return results

def video_list_txt(txt_path):
    """
    :param txt_path:
    :return:
    """
    results = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            video_name = line[0]
            tmp = results.get(video_name, [])
            tmp.append(line[1])
            results[video_name] = tmp
    return results

def find_file_path_by_name(file_dir, name):
    """

    :param file_dir:
    :param name:
    :return:
    """
    file_info = os.popen('find {} -name "*{}*"'.format(file_dir, name)).readlines()
    results = [file.strip() for file in file_info]
    return results


def main():
    txt_path = 'TB_label_24.txt'
    labels = read_label_by_txt(txt_path)
    video_dir = '/mnt/data/Error_Analysis'
    save_dir = '/mnt/data/label_24'
    os.makedirs(save_dir, exist_ok=True)
    for video_name in tqdm(list(labels.keys())):
        video_paths = find_file_path_by_name(video_dir, video_name)
        for video_path in video_paths:
            shutil.copy(video_path, save_dir)


if __name__ == '__main__':
    main()
