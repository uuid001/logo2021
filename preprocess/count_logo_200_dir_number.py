# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/12/21 8:29 PM
# @Version	1.0
# --------------------------------------------------------
import os
import random

# root_dir = '/mnt/ai_data/TB/imgs'
# root_dir = '/mnt/data/logo/training_200_ori'
root_dir = '/mnt/data/TB/anno_make_watermark_base'

classify_number = 200

loss_num = 0
less_num = 0
total_num = 0
for i in range(classify_number):
    classify_path = os.path.join(root_dir, str(i + 1))
    if os.path.exists(classify_path):
        json_files = [file_name for file_name in os.listdir(classify_path) if file_name.endswith('json')]
        for json_file in json_files[::-1]:
            json_suffix = json_file.split('.')
            json_suffix[-1] = 'jpg'
            img_name = '.'.join(json_suffix)
            img_path = os.path.join(classify_path, img_name)
            if not os.path.exists(img_path):
                json_suffix[-1] = 'png'
                img_name = '.'.join(json_suffix)
                img_path = os.path.join(classify_path, img_name)
            if not os.path.exists(img_path):
                json_suffix[-1] = 'jpeg'
                img_name = '.'.join(json_suffix)
                img_path = os.path.join(classify_path, img_name)
            if not os.path.exists(img_path):
                json_file = os.path.join(classify_path, json_file)
                command = 'rm -rf {}'.format(json_file)
                print(command)
                os.system(command)

        dir_num = len(json_files)
        # k = 600
        # if dir_num > k:
        #     print("Dir: {}, Number: {}".format(i + 1, dir_num))
        #     less_num += 1
        print("Dir: {}, Number: {}".format(i + 1, dir_num))
        total_num += dir_num
    else:
        print("Don't have the {} classify.".format(i + 1))
        loss_num += 1

print(loss_num)
print(less_num)
print(total_num / (200 - loss_num))
