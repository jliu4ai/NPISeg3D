#!/usr/bin/env bash

python eval_multi_obj.py --dataset_mode=multi_obj \
               --scan_folder=../datasets/PartNet/scans \
               --val_list=../datasets/PartNet/val_list.json \
               --output_dir=results/PartNet_multi \
               --checkpoint=output/2024-12-18-16-05-25/best_checkpoint.pth \
               --scene_weight=0.05 