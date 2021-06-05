# coding:utf-8
import torch
import cv2
import os
import numpy as np
import time
import re

# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
os.sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'jobs/cv_common'))
os.sys.path.insert(0, '../yolov5')
from detection.postprocess.nms import non_max_suppression, IoU
from detection.preprocess.image import letterbox

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')


def map_data():
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
    return {'ori_id': ori_index, 'new_id': new_index, 'class_id': categories}


def visual_image_bbox(img, bbox, names=None):
    """

    :param img:
    :param bbox: [[x1, y1, x2, y2, label], ...]
    :param names: type: dict, e.g: {1: 'A', 2: 'B'}
    :return:
    """
    for box in bbox:
        x1, y1, x2, y2, label = box[:5]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
        if names is not None:
            name = names[int(label)]
            name = re.sub('_\d+', '', name)
            cv2.putText(img, name, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    return img


def pre_img(model, img, fp16=True, int8=False, debug=False, device='cuda:0'):
    conf_thres = 0.75
    iou_thres = 0.45
    img = np.array(img)
    img = img.reshape(-1, *img.shape)
    img /= 255.0
    if debug:
        conf_thres = 0.1
    if int8:
        output, classid, score = model([img], conf_thres, iou_thres)
        # print(score)
        if len(classid):
            output = np.hstack((output, np.array(score).reshape((-1, len(classid))).T,
                                np.array(classid).reshape((-1, len(classid))).T))
    else:
        img = torch.from_numpy(img).cuda().float()
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.to(device, non_blocking=True)
        if fp16:
            img = img.half()
        with torch.no_grad():
            # img = img.permute(0, 2, 3, 1)
            pre, feature = model(img)
            lb = []
            output = non_max_suppression(pre, conf_thres, iou_thres, labels=lb)[0].cpu().numpy()
    return output


def load_model(model_path, plugin_path=None, device='cuda:0'):
    if plugin_path is not None:
        from yolov5_trt import init_yolov5_trt

        model = init_yolov5_trt(model_path, plugin_path)
    else:
        model = torch.load(model_path, map_location=device)['model'].float().fuse().eval()
        model = model.half()
    return model


def image_process(img, img_size):
    # h0, w0 = img.shape[:2]  # orig hw
    # r = img_size / max(h0, w0)  # resize image to img_size
    # if r != 1:  # always resize down, only resize up if training with augmentation
    #     interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    #     img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    # img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]

    # img, ratio, pad = letterbox(img, 640)
    img = np.ascontiguousarray(img[:, :, ::-1])
    img = img.astype(np.float32)
    return img, (h, w), 0, 0  # img, hw_resized, ratio_padding


def data_preprocess(img_path, label_path):
    img_size = 640
    img = cv2.imread(img_path)
    img, (h, w), ratio, pad = image_process(img, img_size)

    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            labels.append([float(i) for i in line])
    x = np.array(labels).astype(np.float32)
    labels = x.copy()
    labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
    labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
    labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
    labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

    # for out in labels:
    #     cv2.rectangle(img, (int(out[1]), int(out[2])), (int(out[3]), int(out[4])), (0, 255, 0))
    # cv2.imwrite('test1.jpg', img)
    return img, labels


def items_image(img_dir, target_dir, debug=False):
    for target in os.listdir(target_dir):
        if not target.endswith('.txt'):
            continue
        target_path = os.path.join(target_dir, target)
        target_name = target.replace('.txt', '')
        img_name = '{}.jpg'.format(target_name)
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, '{}.png'.format(target_name))
        assert os.path.exists(img_path)

        img, label = data_preprocess(img_path, target_path)
        if debug:
            yield img, label, img_name
        else:
            yield img, label


def compute_pr(model_path, img_dir, label_dir):
    model = load_model(model_path)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    draw_dir = os.path.abspath('draw')
    non_dir = os.path.abspath('non')
    os.system('rm -rf {} && rm -rf {}'.format(draw_dir, non_dir))
    os.makedirs(draw_dir, exist_ok=True)
    os.makedirs(non_dir, exist_ok=True)
    tp = 0
    pre_total = 0
    tar_total = 0
    start_time = time.time()
    for img, labels, img_name in items_image(img_dir, label_dir, debug=True):
        output = pre_img(model, img)

        tar_total += len(labels)
        pre_total += len(output)

        for label in labels:
            if not len(output):
                continue
            out = output[output[:, 5] == label[0]]
            if not len(out):
                continue
            ious = IoU(np.array([label[1:]]), out[:, :4]).max()
            if ious > 0.5:
                tp += 1
            else:
                print(23)

        if len(output):
            for out in output:
                cv2.rectangle(img, (out[0], out[1]), (out[2], out[3]), (255, 0, 0))
            # for label in labels:
            #     cv2.rectangle(img, (label[1], label[2]), (label[3], label[4]), (0, 0, 255))
            cv2.imwrite(os.path.join(draw_dir, img_name), img)
        else:
            cv2.imwrite(os.path.join(non_dir, img_name), img)
        # break
    # torch.save(model, 'new_model.pt')
    end_time = time.time()
    p = tp / pre_total
    r = tp / tar_total
    print("P: {}, R: {}, Cost time: {}".format(p, r, (end_time - start_time)))


def predict(model_path, img_dir, plugin_path=None):
    model = load_model(model_path, plugin_path=plugin_path)
    draw_dir = os.path.abspath('draw')
    non_dir = os.path.abspath('non')
    os.system('rm -rf {} && rm -rf {}'.format(draw_dir, non_dir))
    os.makedirs(draw_dir, exist_ok=True)
    os.makedirs(non_dir, exist_ok=True)
    tp = 0
    pre_total = 0
    tar_total = 0
    start_time = time.time()
    for i, img_name in enumerate(os.listdir(img_dir)):
        if img_name.split('.')[-1] not in ['png', 'jpg', 'jpeg']:
            continue
        img_path = os.path.join(img_dir, img_name)
        img_size = 640
        img = cv2.imread(img_path)
        img, (h, w), ratio, pad = image_process(img, img_size)
        output = pre_img(model, img, int8=True if plugin_path is not None else False)

        if len(output):
            for out in output:
                cv2.rectangle(img, (out[0], out[1]), (out[2], out[3]), (255, 0, 0))
            # for label in labels:
            #     cv2.rectangle(img, (label[1], label[2]), (label[3], label[4]), (0, 0, 255))
            cv2.imwrite(os.path.join(draw_dir, img_name), img)
        else:
            cv2.imwrite(os.path.join(non_dir, img_name), img)
        # break
    # torch.save(model, 'new_model.pt')
    end_time = time.time()
    print("Cost time: {}".format((end_time - start_time)))


def test_single_image(img_path, model_path, plugin_path=None):
    img_size = 640
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = img_path
    img, (h, w), _, _ = image_process(img, img_size)

    model = load_model(model_path, plugin_path)

    output = pre_img(model, img, int8=True if plugin_path is not None else False)

    draw_dir = os.path.abspath('draw')
    non_dir = os.path.abspath('non')
    os.system('rm -rf {} && rm -rf {}'.format(draw_dir, non_dir))
    os.makedirs(draw_dir, exist_ok=True)
    os.makedirs(non_dir, exist_ok=True)
    img_name = os.path.basename(img_path)
    if len(output):
        for out in output:
            cv2.rectangle(img, (out[0], out[1]), (out[2], out[3]), (255, 0, 0))
            cv2.putText(img, str(out[4]), (int(out[0]), int(out[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)
        # for label in labels:
        #     cv2.rectangle(img, (label[1], label[2]), (label[3], label[4]), (0, 0, 255))
        cv2.imwrite(os.path.join(draw_dir, img_name), img)

    else:
        cv2.imwrite(os.path.join(non_dir, img_name), img)


def main():
    # model_path = '/home/lintao/jobs/training/logo/yolov5/runs/train/exp22/weights/best.pt'
    # model_path = '/home/lintao/jobs/training/logo/yolov5/runs/train/exp32/weights/last.pt_242.pt'
    # model_path = '/home/lintao/jobs/training/logo/yolov5/runs/train/exp41/weights/last.pt_0.pt'
    # plugin_path = None
    plugin_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/libmyplugins.so"
    # model_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/best_32_int8.engine"
    model_path = "/home/lintao/jobs/training/logo/tensorrtx/yolov5/build/best_0.engine"
    # img_val_dir = '/mnt/data/logo/training_small_test/images/val'
    # label_val_dir = '/mnt/data/logo/training_small_test/labels/val'
    # img_dir = '/home/lintao/jobs/training/logo/postprocess/test_images'
    # img_dir = '/mnt/data/logo/training_200_ori/34'
    # img_dir = '/home/lintao/jobs/training/logo/postprocess/test_images'
    img_dir = '/home/lintao/jobs/training/logo/postprocess/test_images/yuv_dir/5e9ac2_5e9ac2_1088x720_2.yuv'
    predict(model_path, img_dir, plugin_path=plugin_path)
    # img_path = './test_images/test_11.png'
    # img_path = '/mnt/data/logo/training_200_ori/34/34_48_0.png'
    # img_path = '/home/lintao/jobs/training/logo/postprocess/test_images/1_caricatures_normal_8483.jpg'
    # test_single_image(img_path, model_path, plugin_path=plugin_path)


if __name__ == '__main__':
    main()
