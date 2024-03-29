# Defect detection from images using the ARMBench Dataset
## Introduction

This repository sample scripts to train and test an image-based defect detection model using PyTorch.

## Requirements

Download the Defect Detection (images) dataset from [armbench.com](http://armbench.com)

Extract armbench-defects-image-0.1.tar.gz to filesystem 

```
tar -xf armbench-defects-image-0.1.tar.gz

export AB_DD_IMAGES_DATA=`pwd`
```


Setup a python environment:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Test a pre-trained model on a tiny dataset
```
wget https://armbench-dataset.s3.amazonaws.com/defects/armbench-defects-tiny.tar.gz
wget https://armbench-dataset.s3.amazonaws.com/defects/defects-image/defect_detection_images.ckpt
python train.py --mode test --dataset_path $TINY_DATASET_PATH --checkpoint $CHECKPOINT
```

## Convert the dataset to Imagenet format
```
python convert_to_imagenet.py --dataset_path $AB_DD_IMAGES_DATA
```
## Train a model for the first time
```
python train.py --dataset_path $AB_DD_IMAGES_DATA
```
## To continue training a model
```
python train.py --dataset_path $AB_DD_IMAGES_DATA --checkpoint output/last.ckpt --resume
```
## Test a model
```
python train.py --mode test --dataset_path $AB_SEG_DATA --checkpoint output/last.ckpt
```
