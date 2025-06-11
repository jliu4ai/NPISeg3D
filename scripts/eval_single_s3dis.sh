#!/usr/bin/env bash

python eval_single_obj.py --dataset=s3dis \
               --dataset_mode=single_obj \
               --scan_folder=../datasets/S3DIS/single/crops \
               --crop \
               --val_list=../datasets/S3DIS/single/object_ids.npy \
               --val_list_classes=../datasets/S3DIS/single/object_classes.txt \
               --output_dir=results/S3DIS_single \
               --checkpoint=output/2024-12-18-16-05-25/best_checkpoint.pth \
               --scene_weight=0.1 \
               --cubeedge=-1