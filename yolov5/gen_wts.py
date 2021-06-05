import torch
import struct
import sys
from utils.torch_utils import select_device
import os
# Initialize
device = select_device('cpu')
# pt_file = sys.argv[1]
pt_file = '/home/lintao/jobs/training/logo/yolov5/runs/train/exp/weights/best.pt'
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32
model.to(device).eval()

with open('.'.join(os.path.basename(pt_file).split('.')[:-1]) + '_0.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
