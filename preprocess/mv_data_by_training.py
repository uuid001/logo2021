# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/12/21 8:51 PM
# @Version	1.0
# --------------------------------------------------------
import os
import random
import shutil

root_dir = '/mnt/data/logo/training_200_ori'
rm_dir1 = '/mnt/data/TB/rm_dir1'
rm_dir2 = '/mnt/data/TB/rm_dir2'
training_img_dir = '/mnt/data/logo/training/images/train'
training_json_dir = '/mnt/data/logo/training/labels/train'

classify_number = 200

loss_num = 0
less_num = 0
total_num = 0
for i in range(classify_number):
    classify_path = os.path.join(root_dir, str(i + 1))
    if os.path.exists(classify_path):
        files = os.listdir(classify_path)
        dir_num = len([file_name for file_name in files if file_name.endswith('json')])
        k = 600
        if dir_num > k:
            print("Dir: {}, Number: {}".format(i + 1, dir_num))
            rm_files = random.sample(os.listdir(classify_path), len(files) - k)
            rm_dir_tmp1 = os.path.join(rm_dir1, str(i + 1))
            rm_dir_tmp2 = os.path.join(rm_dir2, str(i + 1))
            os.makedirs(rm_dir_tmp1, exist_ok=True)
            os.makedirs(rm_dir_tmp2, exist_ok=True)
            for rm_file in rm_files:
                try:
                    file_name = '.'.join(rm_file.split('.')[:-1])
                    rm_files = [os.path.join(classify_path, file) for file in files if
                                file_name == '.'.join(file.split('.')[:-1])]
                    for tmp in rm_files:
                        shutil.move(tmp, rm_dir_tmp1)
                        # print(123)

                    for tmp in os.listdir(training_img_dir):
                        if file_name == '.'.join(tmp.split('.')[:-1]):
                            img_path = os.path.join(training_img_dir, tmp)
                            json_path = os.path.join(training_json_dir, '{}.txt'.format(file_name))
                            shutil.move(img_path, rm_dir_tmp2)
                            shutil.move(json_path, rm_dir_tmp2)
                except Exception as e:
                    continue

print(loss_num)
print(less_num)
print(total_num / (200 - loss_num))
