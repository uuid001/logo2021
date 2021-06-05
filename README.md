# logo2021
Logo Detection

## Yolov5
> Reference: https://github.com/ultralytics/yolov5
### Training steps:
1. transform json to txt: <br>
  a. git clone git@github.com:scistor001/cv_common.git <br>
  b. cd cv_common/detection/data_process <br>
  c. run the 'labelme2txt.py' script. <br>
2. config the parameters:  <br>
  a. hyper-parameter profile: yolov5/data/hyp.scratch.yaml <br>
  b. training-data profile: yolov5/data/logo_197.yaml  <br>
  c. model profile: yolov5/models/yolov5s_logo_197.yaml <br>
3. training command: <br>
  nohup python train.py --data ./data/logo_197.yaml --cfg ./models/yolov5s_logo_197.yaml --weights ./runs/train/exp3/weights/best.pt --epochs 200 --batch-size 120 &

## Tensorrtx
> Reference: https://github.com/wang-xinyu/tensorrtx
