#!/usr/bin/env bash

python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=../datasets/KITTI360/scans \
               --val_list=../datasets/KITTI360/val_list.json \
               --output_dir=results/KITTI360_multi \
               --checkpoint=output/2024-12-18-16-05-25/best_checkpoint.pth 