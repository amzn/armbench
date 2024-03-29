#!/bin/bash

if [[ -v AB_DD_VIDEOS_DATA ]]; 
then 

python -m src.train \
    --model "MViT_B" \
    --train_video_label_file "$AB_DD_VIDEOS_DATA/train.csv" \
    --val_video_label_file "$AB_DD_VIDEOS_DATA/test.csv" \
    --dataset_path "$AB_DD_VIDEOS_DATA" \
    --num_classes 2 \
    --train_spatial_size 224 \
    --test_spatial_size 224 \
    --num_samples_per_clip 32 \
    --temporal_size 32 \
    --lr 0.00003 \
    --weight_decay 0.02 \
    --max_epochs 30 \
    --workers 8 \
    --batch_size 2 \

else
echo "Need to set $$AB_DD_VIDEOS_DATA to dataset location"
fi