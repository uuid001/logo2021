# coding:utf-8
import difflib
import os
from collections import Counter
from sklearn.metrics import classification_report
import re
import json

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.expanduser('~'), 'jobs')))
os.sys.path.insert(0, os.path.abspath(os.path.join('..', 'yolov5')))
from cv_common.video.video_reader import VideoReader
from test_images import pre_img, load_model, image_process


def set_video_var(video_reader, video_path):
    video_reader.reset_video_path(video_path)
    # fps = video_reader.get_fps()
    # video_reader.set_total_frames(fps * 60 * 1)
    # video_reader.set_frame_interval(40)
    return video_reader


def map_index(index):
    ori_index = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
                 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
                 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    new_index = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 23, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41,
                 42, 43, 44, 46, 47, 50, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 66, 67, 68, 71, 72, 73, 77, 78, 79, 80,
                 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 99, 102, 104, 105, 106, 107, 108, 109, 110,
                 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                 132, 133, 134, 135, 136, 137, 139, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
                 152, 154, 155, 156, 158, 159, 160, 161, 162, 163, 164, 165, 167, 168, 169, 171, 172, 173, 174, 176,
                 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196,
                 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
                 218, 219, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
                 239]

    categories = ['1', '2', '3', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                  '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
                  '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53',
                  '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
                  '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87',
                  '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '102', '103', '104',
                  '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118',
                  '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132',
                  '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146',
                  '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160',
                  '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174',
                  '175', '176', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189',
                  '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200']

    return new_index[ori_index.index(int(categories[index]))]
    # return categories[index]


import cv2


def predict(model, video_reader, video_path, int8=False):
    device = 'cuda:0'

    video_reader = set_video_var(video_reader, video_path)

    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    results = {}
    ret = True
    while ret:
        ret, frame = video_reader.items_key_frame()
        if frame is None:
            continue
        frame, *_ = image_process(img=frame, img_size=640)
        output = pre_img(model, img=frame, int8=int8, device=device)
        for bbox in output[:1]:
            label = int(bbox[-1])
            # name = names.get(label)
            name = str(map_index(label))
            results[video_reader.get_current_frame()] = name
    results = sorted(results.items(), key=lambda kv: kv[0])
    print("Result: {}".format(results))
    return results


def compute_pr(preds, targets):
    """

    :param :
    :return:
    """
    tp = 0
    fp = 0
    flags = [1] * len(targets)
    error_flags = [0] * len(targets)
    for pred in preds:
        curr_frame, pre_label = int(pred[0]), int(pred[1])
        for i, target in enumerate(targets):
            if target[0] <= curr_frame <= target[1]:
                if pre_label == target[2]:
                    tp += 1
                else:
                    error_flags[i] = 1
                    fp += 1
                flags[i] = 0

    for i, target in enumerate(targets):
        if target[-1] == 0 and flags[i] == 1:
            flags[i] = 0
    new_result = []
    for f, e in zip(flags, error_flags):
        if f or e:
            new_result.append(1)
        else:
            new_result.append(0)
    # print("TP: {}, FP:{}, P: {}, Fragment: LOSS: {}, P: {}".format(tp, fp, (float(tp) / (tp + fp)), sum(new_result),
    #                                                                1 - sum(new_result) / len(new_result)))
    acc = 1 - sum(new_result) / len(new_result)
    print("Fragment: LOSS: {}, Error: {}, Acc: {}, Loss Data: {}, Error Data: {}".format(sum(flags), sum(error_flags),
                                                                                         acc, flags, error_flags))
    return sum(flags), tp, fp, acc


def args_file(file_path):
    result = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line_info = line.strip().split('|')
            result.append((int(line_info[4]), int(line_info[5]), int(line_info[-1])))
    return result


def classify_video_test():
    # with open('predict_result_1.json', 'r') as f:
    #     predict_result = json.load(f)
    # compute_pr(predict_result)
    # exit(0)

    # model_path = '/home/lintao/jobs/training/logo/yolov5/runs/train/exp22/weights/last.pt_110.pt'
    # model_path = '/home/lintao/jobs/training/logo/yolov5/runs/train/exp32/weights/best.pt'
    # plugin_path = None
    plugin_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/libmyplugins.so"
    model_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/best_32_int8.engine"
    video_dir = '/mnt/data/rmzk/TB'
    label_dir = '/mnt/data/rmzk/labels'
    # video_dir = '/mnt/data1/TB/Total_Video/China'
    model = load_model(model_path, plugin_path)
    video_reader = VideoReader(frame_interval=30)
    total_acc = []
    tps = []
    fps = []
    losses = []
    for i, video_file in enumerate(os.listdir(video_dir)):
        # if i != 0:
        #     continue
        # if video_file != 'T_A999996.ts':
        #     continue
        print("========================================")
        print(video_file)
        video_path = os.path.join(video_dir, video_file)
        try:
            results = predict(model, video_reader, video_path,
                              int8=False if plugin_path is None else True)
            labels = args_file(os.path.join(label_dir, '{}.FileInfo'.format('.'.join(video_file.split('.')[:-1]))))
            loss, tp, fp, acc = compute_pr(results, labels)
            total_acc.append(acc)
            tps.append(tp)
            fps.append(fp)
            losses.append(loss)
        except Exception as e:
            print("video path: {} has error: {}".format(video_path, e))
    print("Avg Acc: {}".format(sum(total_acc) / len(total_acc)))
    print("Avg P: {}".format(sum(tps) / (sum(tps) + sum(fps))))
    print("AVg R: {}".format(1 - sum(losses) / (sum(tps) + sum(fps) + sum(losses))))


if __name__ == '__main__':
    classify_video_test()
