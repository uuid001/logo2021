# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 6/3/21 8:15 PM
# @Version	1.0
# --------------------------------------------------------
import os
os.sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'jobs'))
import shutil
from functools import partial
from cv_common.threads.mutli_process import MultiProcess


def read_txt(txt_path):
    results = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            class_id, img = line.strip().split(',')
            tmp = results.get(class_id, [])
            tmp.append(img)
            results[class_id] = tmp
    return results


def copy_data(data, results):
    src_dir = '/mnt/data/logo/training_200_ori'
    dst_dir = '/mnt/data/TB/anno_make_watermark_base'
    for class_id in data:
        images = results.get(class_id)
        for image in images:
            try:
                src_img_path = os.path.join(src_dir, class_id, image)
                src_json_path = os.path.join(src_dir, class_id, '{}.json'.format('.'.join(image.split('.')[:-1])))
                assert os.path.exists(src_img_path) and os.path.exists(src_json_path)
                new_dir = os.path.join(dst_dir, class_id)
                os.makedirs(new_dir, exist_ok=True)
                if os.path.exists(os.path.join(new_dir, image)):
                    continue
                shutil.copy(src_img_path, new_dir)
                shutil.copy(src_json_path, new_dir)
            except Exception as e:
                print("{}: {}: {}".format(src_img_path, src_json_path, e))


def main():
    txt_path = '/mnt/data/TB/anno_make_watermark_base/result.txt'
    data = read_txt(txt_path)
    func = partial(copy_data, results=data)
    mp = MultiProcess(20)
    mp.set_job(func, data=list(data.keys()))
    mp.start()


if __name__ == '__main__':
    main()
