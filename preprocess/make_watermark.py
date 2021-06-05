import json
import os
import random
import uuid
from functools import partial

import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm

os.sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'jobs'))
os.sys.path.insert(0, '..')
from cv_common.threads.mutli_process import MultiProcess
from common.img import random_brightness, random_contrast


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


def make_json_object(label, i_w, i_h, x1, y1, x2, y2, img_path=None):
    data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [
            {
                "label": "{}".format(label),
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": img_path,
        "imageData": None,
        "imageHeight": i_h,
        "imageWidth": i_w
    }
    return data


def dilate_erosion(img):
    if random.randint(0, 1):
        # dilation
        img = img.filter(ImageFilter.MaxFilter(3))
    else:
        # erosion
        img = img.filter(ImageFilter.MinFilter(3))
    return img


def random_brightness_contrast(img):
    ratio = random.uniform(0.8, 1.2)
    img = random_brightness(img, ratio)
    if random.randint(0, 1):
        ratio = random.uniform(0.8, 1.2)
        img = random_contrast(img, ratio)
    return img


def scale_background_img(input_img):
    """

    :param input_img: input path
    :return:
    """
    # the max slide should be lower than 3072
    i_input = cv2.imread(input_img)
    i_h, i_w, i_c = i_input.shape
    i_r = 3072.0 / max(i_h, i_w)
    if i_r < 1:
        i_h = int(i_h * i_r)
        i_w = int(i_w * i_r)
        cv2.imwrite(input_img, cv2.resize(i_input, (i_w, i_h)))

    # the min slide should be lower than 3072
    i_r = 320.0 / min(i_h, i_w)
    if i_r > 1:
        i_h = int(i_h * i_r)
        i_w = int(i_w * i_r)
        cv2.imwrite(input_img, cv2.resize(i_input, (i_w, i_h)))
    return i_w, i_h


def random_resize_logo(img):
    l_w, l_h = img.size
    if random.randint(0, 1):
        r = random.uniform(0.5, 1.5)
        if random.randint(0, 1):
            l_w *= r
        else:
            l_h *= r

    l_r = l_w / l_h
    if random.randint(0, 1):
        r = random.uniform(0.3, 3)
        l_r *= r
        if r > 1:
            if l_w > l_h:
                l_w = l_h * l_r
            else:
                l_h = l_w * l_r
        else:
            if l_w > l_h:
                l_w = l_h * l_r
            else:
                l_h = l_w * l_r
    return l_w, l_h


def ffmpeg_single_img(input_logo_path, input_img_path, output_img):
    """

    :param input_logo_path:
    :param input_img_path:
    :param output_img:
    :return:
    """

    def save_new_log(img):
        new_tmp_dir = '/mnt/data/tmp'
        os.makedirs(new_tmp_dir, exist_ok=True)
        new_path = os.path.join(new_tmp_dir, os.path.basename(input_logo_path))
        img.save(new_path)
        return new_path

    l_input = Image.open(input_logo_path)

    # random dilate and erosion for logo
    if random.uniform(0, 1) < 0:
        l_input = dilate_erosion(l_input)
        input_logo_path = save_new_log(l_input)

    # random gaussian blur
    if random.uniform(0, 1) < 0:
        # l_input = l_input.filter(ImageFilter.GaussianBlur(radius=3))
        l_input = l_input.filter(ImageFilter.GaussianBlur(radius=2))
        input_logo_path = save_new_log(l_input)

    # random contrast and brightness for logo
    if random.uniform(0, 1) < 0.3:
        l_input = Image.open(input_logo_path)
        l_input = random_brightness_contrast(l_input)
        input_logo_path = save_new_log(l_input)

    # random resize logo:
    l_w, l_h = random_resize_logo(l_input)

    # scale background image:
    i_w, i_h = scale_background_img(input_img_path)

    # assert:
    if l_w > i_w - 5 or l_h > i_h - 5:
        raise Exception

    # merge background image and logo
    start_w = int(i_w - l_w)
    start_h = int(i_h - l_h)
    x = random.randint(min(5, start_w), max(5, start_w))
    y = random.randint(min(5, start_h), max(5, start_h))
    command = "ffmpeg -y -v quiet " \
              "-i {} " \
              "-vf 'movie={},scale={}:{} [logo];[in][logo] " \
              "overlay={}:{} [out]' {}".format(input_img_path, input_logo_path, l_w, l_h, x, y,
                                               output_img)
    os.system(command)

    # generate json file about target
    logo_label = '.'.join(os.path.basename(input_logo_path).split('.')[:-1]).split('_')[0]
    json_data = make_json_object(logo_label, i_w, i_h, x, y, x + l_w, y + l_h, os.path.basename(output_img))
    json_dir = os.path.dirname(output_img)
    os.makedirs(json_dir, exist_ok=True)
    with open(os.path.join(json_dir, '{}.json'.format('.'.join(os.path.basename(output_img).split('.')[:-1]))),
              'w') as f:
        json.dump(json_data, f)


def compute_number_per_logo(logo_inds, logos):
    ori_dir = os.path.join('/mnt/data/TB/anno_make_watermark_base', logo_inds)
    total_num = 1000 - (len(os.listdir(ori_dir)) if os.path.exists(ori_dir) else 0) / 2
    ori_ratio = 0.2

    ori_len = len([logo for logo in logos if 'ori' in logo])
    refine_len = len(logos) - ori_len
    ori_num_per = int((total_num * ori_ratio) / ori_len) if ori_len else 0
    refine_num_per = int((total_num - ori_num_per) / refine_len) if refine_len else 1
    residual_num = total_num - (ori_num_per * ori_len + refine_num_per * refine_len)
    result = []
    for i, logo in enumerate(logos):
        if 'ori' in logo:
            result.append((logo, ori_num_per))
        else:
            if i == len(logos) - 1:
                result.append((logo, refine_num_per + residual_num))
            else:
                result.append((logo, refine_num_per))
    return result


def generate_logo(logo_inds, logo_data, img_paths, now_json_number, save_dir):
    for logo_ind in tqdm(logo_inds):
        logos = logo_data.get(logo_ind, [])
        if len(logos) == 0:
            continue
        logo_info = compute_number_per_logo(logo_ind, logos)
        for logo_path, per_imgs in logo_info:
            for _ in range(int(per_imgs)):
                flag = True
                while flag:
                    try:
                        img_path = random.choice(img_paths)
                        # logo_path = random.choice(logos)
                        new_name = '{}_{}_{}'.format(str(uuid.uuid1())[:8], logo_ind, os.path.basename(img_path))
                        new_save_dir = os.path.join(save_dir, logo_ind)
                        os.makedirs(new_save_dir, exist_ok=True)
                        output_video = os.path.join(new_save_dir, new_name)
                        ffmpeg_single_img(logo_path, img_path, output_video)
                        flag = False
                    except Exception as e:
                        print(e)


def now_logo_json():
    json_dir = '/mnt/data/TB/annotation/jsons'
    result = {}
    for label in os.listdir(json_dir):
        label_path = os.path.join(json_dir, label)
        result[label] = len(os.listdir(label_path))
    return result


def generate_logo_video_multi_threads():
    # now_json_number = now_logo_json()
    now_json_number = {}

    # save_dir = './logo/test'
    save_dir = '/mnt/data/TB/anno_make_watermark_base'
    logo_dir = '/mnt/data/TB/TB_logo'
    # img_dir = '/mnt/data/TB/background'
    img_dir = '/mnt/data/TB/real_normal'
    imgs = []
    all_files(img_dir, imgs)

    logo_data = count_logo_data(logo_dir)
    # for i in range(200):
    #     print("I: {}, Result: {}".format(i + 1, logo_data.get(str(i + 1), None)))
    # print(123)
    mp = MultiProcess(20)
    func = partial(generate_logo, logo_data=logo_data, img_paths=imgs, now_json_number=now_json_number,
                   save_dir=save_dir)
    mp.set_job(func, logo_inds=list(logo_data.keys()))
    mp.start()


def main():
    generate_logo_video_multi_threads()


if __name__ == '__main__':
    main()
