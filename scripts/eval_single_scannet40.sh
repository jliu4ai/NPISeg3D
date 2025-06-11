#!/usr/bin/env bash

python eval_single_obj.py --dataset=scannet40 \
               --dataset_mode=single_obj \
               --scan_folder=../datasets/ScanNet/scans \
               --val_list=../datasets/ScanNet/single/object_ids.npy \
               --val_list_classes=../datasets/ScanNet/single/object_classes.txt \
               --output_dir=results/ScanNet_single \
               --checkpoint=output/2024-12-18-16-05-25/best_checkpoint.pth \
               --scene_weight=0.2