# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 6/3/21 8:33 PM
# @Version	1.0
# --------------------------------------------------------
# coding:utf-8
import json
import os
from tqdm import tqdm


def all_files(video_path, result, suffix=None):
    if os.path.isfile(video_path):
        if suffix is None:
            result.append(video_path)
        elif video_path.endswith(suffix):
            result.append(video_path)
        return
    for video_name in os.listdir(video_path):
        all_files(os.path.join(video_path, video_name), result, suffix)


def run(json_dir, video_dir):
    assign_multi_label = []
    videos = []
    all_files(video_dir, videos)
    videos = [video for video in videos if not video.endswith('json')]

    jsons = []
    all_files(json_dir, jsons, suffix='json')
    label_num_dict = {}
    flag = True
    for json_file in tqdm(jsons):
        try:
            file_name = os.path.basename(json_file).split('.')[0]
            video_path = None
            for video in videos:
                if file_name == os.path.basename(video).split('.')[0]:
                    video_path = video
                    break
            if video_path is None:
                print("Json don't match the video: {}".format(json_file))
                flag = False
                continue

            dir_label = os.path.basename(os.path.dirname(video_path))
            json_result = json.load(open(json_file, "r", encoding="utf-8"))
            if dir_label not in assign_multi_label and len(json_result['shapes']) > 1:
                print("There are more than 1 rectangle. Path: {}".format(json_file))
                flag = False
                continue

            if dir_label not in assign_multi_label and len(json_result['shapes']) < 1:
                print("There are zero rectangle. Path: {}".format(json_file))
                flag = False
                continue

            if json_result['imageData'] is not None:
                print("ImageData have data. Path: {}".format(json_file))
                flag = False
                continue
            json_label = json_result['shapes'][0]['label'].split('_')[0]
            if dir_label != json_label:
                print("Dir label: {} not match json label: {}. Path: {}".format(dir_label, json_label, json_file))
                flag = False
                continue
            tmp = label_num_dict.get(json_label, 0)
            tmp += 1
            label_num_dict[json_label] = tmp
        except Exception as e:
            print("{} has Error: {}.".format(json_file, e))

    if flag:
        print("Pass Check, Thank You!!!")
    else:
        print("Sorry, Don't Pass!!!")


def main():
    data_dir = '/mnt/data/TB/anno_make_watermark_base'
    for class_id in os.listdir(data_dir):
        json_dir = os.path.join(data_dir, class_id)
        video_dir = os.path.join(data_dir, class_id)
        if not os.path.isdir(json_dir):
            continue
        run(json_dir, video_dir)


if __name__ == '__main__':
    main()
