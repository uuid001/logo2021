# coding:utf-8
import os
import cv2
import numpy as np

os.sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
os.sys.path.insert(0, os.path.abspath(os.path.join('..', 'yolov5')))

from test_images import pre_img, load_model, image_process, visual_image_bbox
from common.video_reader import VideoReader


def smoothness_bbox(buffer):
    index_i = -1
    index_j = -1
    buffer_change = [len(buffer[i + 1][1]) - len(buffer[i][1]) for i in range(len(buffer) - 1)]
    for i, value in enumerate(buffer_change):
        if value <= -1:
            index_i = i
            break
    for j in list(range(len(buffer_change)))[::-1]:
        if buffer_change[j] >= 1:
            index_j = j + 1
            break
    if index_i != -1 and index_j != -1:
        for i in range(index_i, index_j):
            if abs(index_i - i) < abs(index_j - i):
                buffer[i] = (buffer[i][0], buffer[index_i][1])
            else:
                buffer[i] = (buffer[i][0], buffer[index_j][1])


def video_visual(video_reader, video_path, model):
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    video_reader.reset_video_path(video_path)
    fps = video_reader.get_fps()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.basename(video_path)
    cap = cv2.VideoWriter('new_pre_{}'.format(video_name), fourcc, fps, (640, 640), True)
    ret = True
    # delay = fps * 0.5
    delay = fps * 2
    # delay = 10
    buffer = []
    video_reader.set_total_frames(fps * 91)
    video_reader.set_start_frame(50)
    while ret:
        ret, frame = video_reader.items_key_frame()
        if frame is None or video_reader.get_current_frame() in list(range(50 + 374, 50 + 398)):
            continue
        frame, *_ = image_process(img=frame, img_size=640)
        output = pre_img(model, img=frame)
        frame = np.ascontiguousarray(frame[:, :, ::-1])
        buffer.append((frame, output))

        if len(buffer) > delay:
            frame, output = buffer.pop(0)
            if len(output):
                frame = visual_image_bbox(img=frame, bbox=np.delete(output, -2, axis=1), names=names)
            smoothness_bbox(buffer)
            cap.write(frame)

    for frame, output in buffer:
        if len(output):
            frame = visual_image_bbox(img=frame, bbox=np.delete(output, -2, axis=1), names=names)
        smoothness_bbox(buffer)
        cap.write(frame)

    cap.release()


def assign_frame(video_reader, video_path, model, frame_number):
    video_reader.reset_video_path(video_path)
    frame = video_reader.get_img_by_frame(frame_number)
    frame, *_ = image_process(img=frame, img_size=640)
    output = pre_img(model, img=frame)
    frame = np.ascontiguousarray(frame[:, :, ::-1])
    if len(output):
        frame = visual_image_bbox(img=frame, bbox=np.delete(output, -2, axis=1), names=names)


def main():
    video_path = '/home/lintao/datasets/logo_det/demo/【紀元播報】【2020盤點】中國十大新聞 空前災難來襲（上） 大紀元新聞網.mp4'
    # model_path = '/home/lintao/jobs/logo_detect/yolov5/runs/train/exp20/weights/best.pt'
    model_path = '/home/lintao/jobs/logo_detect/yolov5/runs/train/exp20/weights/last.pt'
    model = load_model(model_path)
    video_reader = VideoReader()
    video_visual(video_reader, video_path, model)
    # frame_number = 1135
    # assign_frame(video_reader, video_path, model, frame_number=frame_number)


if __name__ == '__main__':
    main()
