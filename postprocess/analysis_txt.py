# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/29/21 2:02 PM
# @Version	1.0
# --------------------------------------------------------
import os
os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.expanduser('~'), 'jobs')))
from test_images import load_model
from test_video import predict
from cv_common.video.video_reader import VideoReader


def analysis_txt(txt_path, video_dir):
    assert os.path.exists(txt_path)
    videos = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            if 'break' in line:
                break
            if len(line) > 1:
                if 'TB' in line[1]:
                    video_path = os.popen('find {} -name "*{}*"'.format(video_dir, line[0])).read().strip()
                    videos.append(video_path)
            else:
                video_path = os.popen('find {} -name "*{}*"'.format(video_dir, line[0])).read().strip()
                videos.append(video_path)

    return videos


def main():

    txt_path = '/mnt/data/Error_Analysis/total_labels.txt'
    # txt_path = './test_txt/24_videos.txt'
    video_dir = '/mnt/data/Error_Analysis'
    video_paths = analysis_txt(txt_path, video_dir)

    plugin_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/libmyplugins.so"
    model_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/best_32_int8.engine"
    # model_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/best_0.engine"
    model = load_model(model_path, plugin_path)

    video_reader = VideoReader(frame_interval=30)
    for video_path in video_paths:
        print("{}".format(video_path))
        try:
            pre_video_name, count_info = predict(model, video_reader, video_path,
                                                 int8=False if plugin_path is None else True)
            print(pre_video_name)
        except Exception as e:
            print("video path: {} has error: {}".format(video_path, e))


if __name__ == '__main__':
    main()
