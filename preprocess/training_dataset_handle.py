# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 6/2/21 8:02 PM
# @Version	1.0
# --------------------------------------------------------
import os

os.sys.path.insert(0, '../..')
from tqdm import tqdm
import shutil
import json
import random
import cv2
from postprocess.test_images import map_data


def read_txt(label_path):
    with open(label_path, 'r') as f:
        lines = list(f.readlines())
        assert len(lines) == 1, print(label_path)
        line_info = lines[0].strip().split(' ')
    return line_info


def get_img_by_label(img_dir, label_name):
    json_suffix = label_name.split('.')
    json_suffix[-1] = 'jpg'
    img_name = '.'.join(json_suffix)
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        json_suffix[-1] = 'png'
        img_name = '.'.join(json_suffix)
        img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        json_suffix[-1] = 'jpeg'
        img_name = '.'.join(json_suffix)
        img_path = os.path.join(img_dir, img_name)
    assert os.path.exists(img_path)
    return img_path


def count_training_datasets(img_dir, label_dir):
    save_path = 'rm_dir'
    os.makedirs(save_path, exist_ok=True)
    results = {}
    for label_name in tqdm(os.listdir(label_dir)):
        label_path = os.path.join(label_dir, label_name)
        try:
            label = read_txt(label_path)[0]
            tmp = results.get(label, [])
            tmp.append(label_name)
            results[label] = tmp
        except Exception as e:
            print(e)
            img_path = get_img_by_label(img_dir, label_name)
            shutil.move(img_path, save_path)
            shutil.move(label_path, save_path)
    with open('result.json', 'w') as f:
        json.dump(results, f)
    for key, value in results.items():
        print("key: {}, value lenth: {}".format(key, len(value)))


def map_classify_by_assign_label(assign_label):
    data = map_data()
    class_id = data['class_id']
    return str(class_id.index(str(assign_label)))


def random_check_box(img_dir, label_dir, assign_label=None):
    save_dir = 'box_dir'
    os.makedirs(save_dir, exist_ok=True)
    labels = os.listdir(label_dir)
    if assign_label is None:
        labels = random.sample(labels, 1000)

    for label_name in tqdm(labels):
        label_path = os.path.join(label_dir, label_name)
        try:
            label_info = read_txt(label_path)
            if assign_label is not None and label_info[0] != map_classify_by_assign_label(assign_label):
                continue
            img_path = get_img_by_label(img_dir, label_name)
            img = cv2.imread(img_path)
            i_h, i_w, _ = img.shape
            c_x, c_y, w, h = [float(i) for i in label_info[1:]]
            x1 = int((c_x - w / 2) * i_w)
            x2 = int((c_x + w / 2) * i_w)
            y1 = int((c_y - h / 2) * i_h)
            y2 = int((c_y + h / 2) * i_h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.putText(img, label_info[0], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            if assign_label is not None:
                tmp_dir = os.path.join(save_dir, str(assign_label))
                os.makedirs(tmp_dir, exist_ok=True)
                cv2.imwrite(os.path.join(tmp_dir, os.path.basename(img_path)), img)
            else:
                cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)

        except Exception as e:
            print("{} : {}".format(label_path, e))


def main():
    label_dir = '/mnt/data/logo/training/labels/train'
    img_dir = '/mnt/data/logo/training/images/train'
    # count_training_datasets(img_dir, label_dir)
    random_check_box(img_dir, label_dir, assign_label=200)


if __name__ == '__main__':
    main()
