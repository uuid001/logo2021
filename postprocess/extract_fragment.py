# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/7/21 11:58 PM
# @Version	1.0
# --------------------------------------------------------
from postprocess.test_video import load_model, VideoReader, set_video_var, image_process, pre_img, difflib
import os
import json
from tqdm import tqdm


def map_index(index):
    model_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    new_index = [5, 5, 4, 15, 38, 1, 16, 17, 9, 9, 9, 10, 10, 2, 2, 8, 8, -1, -1, 7, 7, 7, -1, -1, 18, -1, -1]
    return new_index[model_index.index(index)]


def all_files(video_path, result):
    if os.path.isfile(video_path):
        result.append(video_path)
        return
    for video_name in os.listdir(video_path):
        all_files(os.path.join(video_path, video_name), result)


def predict(model, video_reader, video_path, target):
    device = 'cuda:0'

    video_reader = set_video_var(video_reader, video_path)

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    results = []
    ret = True
    while ret:
        ret, frame = video_reader.items_key_frame()
        if frame is None:
            continue
        frame, *_ = image_process(img=frame, img_size=640)
        output = pre_img(model, img=frame, device=device)
        if len(output) > 0:
            bbox = output[0]
            label = int(bbox[-1])
            name = names.get(label)
            diff_ratio = difflib.SequenceMatcher(None, name, target).quick_ratio()
            if diff_ratio > 0.3:
                results.append(1)
        else:
            results.append(-1)
    return frames_2_second(results)


def frames_2_second(results):
    interval = []
    tmp = []
    for i, status in enumerate(results):
        if status == 1 and len(tmp) == 0:
            tmp.append(i)
        if status == -1 and len(tmp) == 1:
            tmp.append(i - 1)
            interval.append(tmp)
            tmp = []
    if len(tmp) == 1:
        tmp.append(i)
        interval.append(tmp)
    return interval


def main():
    model_path = '/home/lintao/jobs/logo/yolov5-4.0/runs/train/exp/weights/best.pt'
    # video_dir = '/mnt/mobile_disk_1/logo/total_video/old_logo'
    video_dir = '/run/ai_data/data'
    model = load_model(model_path)
    video_reader = VideoReader()
    json_save_dir = '/run/ai_data/json'
    for video_classify in tqdm(os.listdir(video_dir), desc="Classify: "):
        video_paths = []
        all_files(os.path.join(video_dir, video_classify), video_paths)
        ind = video_classify.split(' ')[0]
        dic = {
            'ID': 'L_{}'.format(ind),
            'videos': []
        }
        for video_path in tqdm(video_paths, desc='Video Path: '):
            r_dic = {
                'video_name': os.path.basename(video_path),
                'appear_ts': [[]]
            }
            video_result = predict(model, video_reader, video_path, target=video_classify)
            r_dic['appear_ts'] = video_result
            dic['videos'].append(r_dic)
        with open(os.path.join(json_save_dir, 'L_{}.json'.format(ind)), 'w') as f:
            json.dump(dic, f)


if __name__ == '__main__':
    main()
