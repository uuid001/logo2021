# coding:utf-8
import difflib
import os
from collections import Counter
from sklearn.metrics import classification_report
import re
import json
import numpy as np
import cv2

# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
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


def predict(model, video_reader, video_path, int8=False, debug=False):
    device = 'cuda:0'

    video_reader = set_video_var(video_reader, video_path)
    # names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    results = []
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
            results.append([name, ])
        if debug:
            print("Frame: {}, Result: {}".format(video_reader.get_current_frame(), output))
    results = Counter(results)
    print(results)
    results = results.most_common(1)
    if len(results) == 0:
        print("================================ {}".format(video_path))
        return '', results
    else:
        return results[0][0], results


def predict_single_frame(model, video_reader, video_path, int8=False, debug=False):
    device = 'cuda:0'
    video_reader = set_video_var(video_reader, video_path)
    results = []
    ret = True
    tb_frame_num = 0
    is_TB = True
    video_frame_num = 1
    if debug:
        new_dir = './draw'
        if os.path.exists(new_dir):
            file_num = new_dir.split('_')[-1]
            if not file_num.isdigit():
                new_dir = '{}_{}'.format(new_dir, '0')
            else:
                new_dir = '{}_{}'.format(new_dir, int(file_num) + 1)
        os.makedirs(new_dir, exist_ok=True)
        new_video_path = os.path.join(new_dir, os.path.basename(video_path))

        size = (640, 640)
        video_writer = cv2.VideoWriter(new_video_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                                       int(video_reader.get_fps()), size)

    while ret:
        ret, frame = video_reader.items_key_frame()
        if frame is None:
            video_frame_num = video_frame_num + 1
            continue
        frame, *_ = image_process(img=frame, img_size=640)
        output = pre_img(model, img=frame, int8=int8, device=device, debug=debug)
        # if len(output[:1]) > 1:
        #     is_TB = False
        if len(output[:1]) == 0:
            results.append([video_frame_num, "0"])
        for bbox in output:
            label = int(bbox[-1])
            # name = names.get(label)
            name = str(map_index(label))
            results.append([video_frame_num, name])
            if debug:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))
                cv2.putText(frame, '{}:{:.2f}'.format(name, bbox[4]), (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
        if debug:
            frame = np.ascontiguousarray(frame[:, :, ::-1]).astype(np.uint8)
            video_writer.write(frame)
        video_frame_num = video_frame_num + 1
    if tb_frame_num > 10:
        is_TB = False
    print("{}  len:".format(video_path), tb_frame_num)
    return results, is_TB


def compute_pr(predict_result):
    """

    :param predict_result: {video_name: [pt, gt, number_info]}
    :return:
    """
    x = []
    y = []
    for video_name, value in predict_result.items():
        pred = re.sub('_\d+', '', value[0]).lower()
        target = value[1].lower()
        diff_ratio = difflib.SequenceMatcher(None, pred, target).quick_ratio()

        if diff_ratio > 0.7:
            x.append(target)
            y.append(target)
        else:
            print("{}: {}".format(video_name, value))
            x.append(pred)
            y.append(target)
    t = classification_report(y, x, labels=list(set(y)))
    print(t)


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
    video_dir = '/mnt/data/TB/Ok'
    # video_dir = '/mnt/data1/TB/Total_Video/China'
    model = load_model(model_path, plugin_path)
    video_reader = VideoReader(frame_interval=30)
    predict_result = {}
    for video_classify in os.listdir(video_dir):
        # if video_classify != '41':
        #     conti_classify != '41':
        #     continue
        video_file_dir = os.path.join(video_dir, video_classify)
        if not os.path.exists(video_file_dir):
            continue
        print(video_file_dir)
        for video_file in os.listdir(video_file_dir):
            # if video_file != 'T_102_07_1.mp4':
            #     continue
            print("===================={}====================".format(video_file))
            video_path = os.path.join(video_file_dir, video_file)
            try:
                pre_video_name, count_info = predict(model, video_reader, video_path,
                                                     int8=False if plugin_path is None else True)
                predict_result[video_file] = [pre_video_name, video_classify.split('_')[0], count_info]
            except Exception as e:
                print("video path: {} has error: {}".format(video_path, e))
    with open('predict_result_int8.json', 'w') as f:
        json.dump(predict_result, f)
    compute_pr(predict_result)


def unrelation_video_test():
    plugin_path = "/home/lintao/jobs/scistor001/logo2021/tensorrtx/build/libmyplugins.so"
    model_path = "/home/lintao/jobs/scistor001/logo2021/tensorrtx/build/best_3_int8.engine"
    video_dir = '/mnt/data/Error_Analysis/TB_25/logo'
    model = load_model(model_path, plugin_path)
    video_reader = VideoReader(frame_interval=0)
    for video_file in os.listdir(video_dir):
        print(video_file)
        # if video_file != 'T_112_07_0008.ts':
        #     continue
        video_path = os.path.join(video_dir, video_file)
        try:
            pre_info = predict_single_frame(model, video_reader, video_path,
                                            int8=False if plugin_path is None else True,
                                            debug=True)
        except Exception as e:
            print("video path: {} has error: {}".format(video_path, e))


# zbw 20210601
def unrelation_video_test_zbw(video_path):
    plugin_path = "/home/zhengbowen/jobs/logo2021/tensorrtx/build/libmyplugins.so"
    model_path = "/home/zhengbowen/jobs/logo2021/tensorrtx/build/best_32.engine"

    model = load_model(model_path, plugin_path)
    video_reader = VideoReader(frame_interval=0)
    try:
        pre_video_result, is_TB = predict_single_frame(model, video_reader, video_path,
                                                       int8=False if plugin_path is None else True)
    except Exception as e:
        print("video path: {} has error: {}".format(video_path, e))
    return pre_video_result, is_TB


def single_video_test():
    plugin_path = "/home/lintao/jobs/scistor001/logo2021/tensorrtx/build/libmyplugins.so"
    model_path = "/home/lintao/jobs/scistor001/logo2021/tensorrtx/build/best_3_int8.engine"
    # model_path = "/home/lintao/jobs/training/logo/yolov5/runs/train/exp32/weights/best.pt"
    # plugin_path = None
    model = load_model(model_path, plugin_path)
    video_reader = VideoReader(frame_interval=0)
    video_path = '/mnt/data/label_24/616b3e5db1a2013a_dba258b2eb9c0348_10.mp4'
    debug = True
    pre_info = predict_single_frame(model, video_reader, video_path,
                                    int8=False if plugin_path is None else True,
                                    debug=debug)
    print(pre_info)


if __name__ == '__main__':
    # classify_video_test()
    unrelation_video_test()
    # single_video_test()
    # unrelation_video_test_zbw("/mnt/data/label_24/80eb8da8c09202be_8df0a52e421a1b97_6.mp4")
