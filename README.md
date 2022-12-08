# Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

This repo contains the code, data and trained models for the paper [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf).

## Overview

MTCNN is a popular algorithm for face detection that uses multiple neural networks to detect faces in images. It is capable of detecting faces under various lighting and pose conditions and can detect multiple faces in an image.

We have implemented MTCNN using the pytorch framework. Pytorch is a popular deep learning framework that provides tools for building and training neural networks. 

![](https://img.enderfga.cn/img/image-20221208152130975.png)

![](https://img.enderfga.cn/img/image-20221208152231511.png)

## Requirements

* numpy
* matplotlib
* opencv-python
* torch

## How to Install

- ```shell
  conda create -n env python=3.8 -y
  conda activate env
  ```
- ```shell
  pip install -r requirements.txt
  ```

## Preprocessing

- download [WIDER_FACE](http://shuoyang1213.me/WIDERFACE/) face detection data then store it into ./data_set/face_detection
- download [CNN_FacePoint](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) face detection and landmark data then store it into ./data_set/face_landmark

### Preprocessed Data

```shell
# Before training Pnet
python get_data.py --net=pnet
# Before training Rnet, please use your trained model path
python get_data.py --net=rnet --pnet_path=./model_store/pnet_epoch_20.pt
# Before training Onet, please use your trained model path
python get_data.py --net=onet --pnet_path=./model_store/pnet_epoch_20.pt --rnet_path=./model_store/rnet_epoch_20.pt
```

## How to Run

### Train

```shell
python train.py --net=pnet/rnet/onet #Specify the corresponding network to start training
bash train.sh                        #Alternatively, use the sh file to train in order
```

The checkpoints will be saved in a subfolder of `./model_store/*`.

#### Finetuning from an existing checkpoint

```shell
python train.py --net=pnet/rnet/onet --load=[model path]
```

model path should be a subdirectory in the `./model_store/` directory, e.g. `--load=./model_store/pnet_epoch_20.pt`

### Evaluate

#### Use the sh file to test in order

```shell
bash test.sh
```

#### To detect a single image

```shell
python test.py --net=pnet/rnet/onet  --path=test.jpg
```

#### To detect a video stream from a camera

```shell
python test.py --input_mode=0
```

#### The result of  "--net=pnet"

![](https://img.enderfga.cn/img/20221208160900.png)

#### The result of  "--net=rnet"

![](https://img.enderfga.cn/img/image-20221208155022083.png)

#### The result of  "--net=onet"

![](https://img.enderfga.cn/img/image-20221208155044451.png)
