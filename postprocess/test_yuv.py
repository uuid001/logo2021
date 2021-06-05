# -*- coding: utf-8 -*- #
# --------------------------------------------------------
# @Author: Lintao
# @Date: 5/19/21 5:40 PM
# @Version	1.0
# --------------------------------------------------------
import cv2
import numpy as np


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = self.width * self.height * 3 / 2
        self.f = open(filename, 'rb')
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:
            raw = self.f.read(int(self.frame_len))
            yuv = np.frombuffer(raw, dtype=np.uint8)
            yuv = yuv.reshape(self.shape)
        except Exception as e:
            print(str(e))
            return False, None
        return True, yuv

    def read(self):
        ret, yuv = self.read_raw()
        if not ret:
            return ret, yuv
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return ret, bgr


if __name__ == "__main__":
    import os

    yuv_dir = './yuv_dir'

    # filename = "./2c40212_2c40212_1088x720.yuv"
    for filename in os.listdir(yuv_dir):
        filename = os.path.join(yuv_dir, filename)
        size = (720, 1088)
        cap = VideoCaptureYUV(filename, size)
        i = 1
        save_path = './test_images/{}'.format(filename)
        if os.path.exists(save_path):
            os.system('rm -rf {}'.format(save_path))
        os.makedirs(save_path, exist_ok=True)

        while i:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(save_path, 'test_{}.png'.format(i)), frame)
            i += 1
