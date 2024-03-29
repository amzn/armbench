# Semantic Segmentation using the ARMBench Dataset
## Introduction

This repository contains a Python notebook to begin loading and viewing the ARMBench semantic segmentation dataset using pycocotools and PyTorch libraries. In addition, it contains sample scripts to train and test a Mask RCNN model using PyTorch.

## Requirements

Download the Segmentation dataset from [armbench.com](http://armbench.com)

Extract armbench-segmentation-0.1.tar.gz to filesystem 

```
tar -xf armbench-segmentation-0.1.tar.gz

cd armbench-segmentation-0.1
export AB_SEG_DATA=`pwd`
```

Download the pretrained mask-rcnn model

```
wget https://armbench-dataset.s3.amazonaws.com/segmentation/latest.pt
```

Setup a python environment:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Train a model for the first time
```
python train_mask_rcnn.py --dataset_path $AB_SEG_DATA
```
## To continue training a model
```
python train_mask_rcnn.py --dataset_path $AB_SEG_DATA --resume-from latest.pt
```
## Test a model
```
python test_mask_rcnn.py --dataset_path $AB_SEG_DATA --resume-from latest.pt 
```

## Run Jupyter notebook
```
jupyter notebook
```
Modify block 2 with the dataset_path ($AB_SEG_DATA). This notebook provides a tutorial on how to load and visualize data using the ARMBench Segmentation dataset.
