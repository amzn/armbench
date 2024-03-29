# Defect detection from videos using the ARMBench Dataset
## Introduction

This repository sample scripts to train and test an video-based defect detection model using PyTorch.

## Requirements

Download the Defect Detection (videos) dataset from [armbench.com](http://armbench.com)

Extract armbench-defects-image-0.1.tar.gz to filesystem 

```
tar -xf armbench-defects-image-0.1.tar.gz

export AB_DD_VIDEOS_DATA=`pwd`
```

Download the required pretrained checkpoint MVIT_B_32x3.pyth
```
mkdir pretrained_ckpt
cd pretrained_ckpt
wget ???
```

Setup a python environment:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Convert the dataset to ImageNet format
```
python scripts/preprocess_video.py --dataset_path $AB_DD_VIDEOS_DATA
```
## Train a model for the first time
```
./train.sh # uses $AB_DD_VIDEOS_DATA env variable for dataset path
```
## Test a model
```
# uses $AB_DD_VIDEOS_DATA env variable for dataset path, update --cpt with checkpoint
./test.sh 
```
