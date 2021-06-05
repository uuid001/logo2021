import os
from tqdm import tqdm

os.sys.path.insert(0, os.path.abspath(os.path.join(os.path.expanduser('~'), 'jobs')))
os.sys.path.insert(0, os.path.abspath(os.path.join('..', 'yolov5')))

from data.extrac_video_by_label import read_label_by_txt, video_list_txt
from postprocess.test_video import predict_single_frame
from cv_common.video.video_reader import VideoReader
from test_images import load_model


def main():
    txt_path = '/home/lintao/jobs/scistor001/logo2021/data/TB_label_24.txt'
    video_dir = "/mnt/data/label_24"
    # txt_path = ' /mnt/data/Error_Analysis/TB_25/logo'
    # video_dir = "/mnt/data/Error_Analysis/TB_25/T/"
    labels = read_label_by_txt(txt_path)

    plugin_path = "/home/lintao/jobs/scistor001/logo2021/tensorrtx/build/libmyplugins.so"
    model_path = "/home/lintao/jobs/scistor001/logo2021/tensorrtx/build/best_3_int8.engine"
    # plugin_path = None
    # model_path = '/home/lintao/jobs/scistor001/logo2021/yolov5/runs/train/exp3/weights/best.pt'
    model = load_model(model_path, plugin_path)
    video_reader = VideoReader(frame_interval=0)
    all_f1 = 0
    all_p = 0
    all_r = 0
    for video_name, video_labels in labels.items():
        tp = 0
        fp = 0
        fn = 0
        # if '9133bcdb9f7d11bd_51b5908d1a8aa32f_32' not in video_name:
        #     continue
        video_path = os.path.join(video_dir, video_name)
        video_results = predict_single_frame(model, video_reader, video_path,
                                             int8=False if plugin_path is None else True,
                                             debug=False)

        # print("video_results:", len(video_results))
        n = 0
        for video_label in video_labels:
            start_frame = int(video_label[0])
            end_frame = int(video_label[1])
            label_class = int(video_label[2])
            # print(end_frame - start_frame + 1)
            for video_result in video_results[::-1]:
                frame_num = int(video_result[0])
                pre_label = int(video_result[1])
                if start_frame <= frame_num <= end_frame:
                    if pre_label == label_class:
                        tp = tp + 1
                    elif pre_label == 0:
                        fn = fn + 1
                    else:
                        fp = fp + 1
                    n = n + 1
                    video_results.remove(video_result)

        for video_result_copy in video_results:
            if int(video_result_copy[1]) != 0:
                fp = fp + 1

        p = tp / (tp + fp) if (tp + fp) else 1
        r = tp / (tp + fn) if (tp + fn) else 1
        f1 = (2 * p * r) / ((p + r) if (p + r) else 1)
        print("Video: {}, P:{:.4f}, R:{:.4f}, F1:{:.4f}".format(video_name, p, r, f1))
        all_f1 += f1
        all_p += p
        all_r += r
    print("Avg: P:{:.4f}, R:{:.4f}, F1:{:.4f}".format(all_p / len(labels), all_r / len(labels), all_f1 / len(labels)))


if __name__ == '__main__':
    # txt_path = '/home/lintao/jobs/scistor001/logo2021/data/TB_label_24.txt'
    # video_dir = "/mnt/data/label_24"
    # pr_count_video(txt_path,video_dir)

    # txt_path_pre = '/home/zhengbowen/jobs/ytj_24_logo.txt'
    # txt_path_true = '/home/zhengbowen/jobs/TB_lablel_24.txt'
    # pr_count_txt(txt_path_pre, txt_path_true)

    txt_path = "/mnt/data/Error_Analysis/total_labels.txt"
