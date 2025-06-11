#!/usr/bin/env bash

python eval_single_obj.py --dataset=kitti360 \
               --dataset_mode=single_obj \
               --scan_folder=../datasets/KITTI360/single/crops \
               --crop \
               --val_list=../datasets/KITTI360/single/object_ids.npy \
               --val_list_classes=../datasets/KITTI360/single/object_classes.txt \
               --output_dir=results/KITTI_single \
               --checkpoint=output/2024-12-18-16-05-25/best_checkpoint.pth \
               --scene_weight=0.1 \
               --cubeedge=0.05