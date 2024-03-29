# Object Identification using the ARMBench Dataset
## Introduction

This repository contains sample scripts to extract DINO embeddings from source and destination images and find a nearest neighbor match.

## Requirements

Download the Tiny Identification dataset from [armbench.com](http://armbench.com)

```
wget https://armbench-dataset.s3.amazonaws.com/identification/armbench-object-id-tiny.tar.gz
```

Extract armbench-object-id-tiny.tar.gz to filesystem 

```
tar -xf armbench-object-id-tiny.tar.gz
cd armbench-object-id-tiny/
export AB_ID_DATA=`pwd`
```


Setup a python environment:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Extract embeddings and get nearest neighbor ID
```
python test.py --dataset_path $AB_ID_DATA
```
## Print ID retreival rate
```
python print_results.py --dataset_path $AB_ID_DATA
```
